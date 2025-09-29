"""Utility classes for pseudo-anomaly mining and score calibration.

The pseudo-anomaly mining strategy implemented here is inspired by recent
semi-supervised anomaly detection literature where a small subset of high score
samples is treated as hard negatives to stabilise the memory bank. In
particular, we follow the spirit of the pseudo outlier exposure strategy from
"Pseudo Outlier Exposure for Industrial Anomaly Detection" (CVPR 2023) and the
robust covariance estimation idea used in "MemSeg: A Semi-supervised Method for
Industrial Anomaly Detection" (CVPR 2022). The contamination-aware patch
filtering stage borrows the adaptive trimming mechanism discussed in
"CleanAD: Pseudo-Label Guided Self-Ensemble for Contaminated Industrial Anomaly
Detection" (CVPR 2024) and the score-tail filtering heuristics from "Selective
Anomaly Detection via Score-Based Filtering" (ICCV 2023) to better withstand
corrupted training data.

The score calibration module adapts the logistic temperature scaling approach
popularised by "On Calibration of Modern Neural Networks" (ICML 2017) to better
separate mined hard negatives from normal samples.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class MiningResult:
    """Container holding pseudo-anomaly mining results.

    The ``normal_features`` field represents the feature tensors that should
    populate the memory bank after any contamination-aware filtering.  It can
    therefore contain contributions from suspected anomalous images once their
    high-scoring patches have been trimmed away.
    """

    normal_features: List[np.ndarray]
    pseudo_anomaly_features: Optional[np.ndarray]
    anomaly_indices: Sequence[int]
    anomaly_scores: np.ndarray
    selection_threshold: Optional[float] = None
    patch_threshold: Optional[float] = None
    rejected_patches: int = 0
    kept_patches: int = 0
    normal_calibration_features: Optional[List[np.ndarray]] = None
    pseudo_calibration_features: Optional[List[np.ndarray]] = None
    separation_gap: Optional[float] = None
    selected_ratio: Optional[float] = None


class PseudoAnomalyMiner:
    """Identify potential anomaly samples within the training split.

    The miner ranks images by their robust Mahalanobis distance and selects the
    top-k fraction as pseudo anomalies. Selected features are removed from the
    normal memory bank and later used for score calibration. The optional
    curriculum-style gap widening is inspired by the progressive relabeling
    heuristics in "Rank & Sort: Score-Based Calibration for Industrial Anomaly
    Detection" (CVPR 2023) and the contamination-aware curriculum discussed in
    "CleanAD: Pseudo-Label Guided Self-Ensemble for Contaminated Industrial
    Anomaly Detection" (CVPR 2024), where the pseudo set is expanded until the
    separation margin between suspected anomalies and retained normals exceeds a
    configurable threshold.
    """

    def __init__(
        self,
        ratio: float = 0.05,
        min_count: int = 10,
        shrinkage: float = 0.05,
        patch_subsample: Optional[int] = 2048,
        seed: Optional[int] = None,
        dynamic_ratio: bool = True,
        tail_quantile: float = 0.995,
        mad_multiplier: float = 3.0,
        max_ratio: Optional[float] = 0.2,
        patch_filter_ratio: float = 0.1,
        patch_filter_subsample: Optional[int] = 512,
        patch_filter_dynamic: bool = True,
        patch_filter_mad_multiplier: float = 3.0,
        dynamic_gap: float = 0.25,
    ) -> None:
        if ratio < 0:
            raise ValueError("ratio must be non-negative")
        self.ratio = ratio
        self.min_count = min_count
        self.shrinkage = shrinkage
        self.patch_subsample = patch_subsample
        self.random_state = np.random.RandomState(seed)
        self.dynamic_ratio = dynamic_ratio
        self.tail_quantile = np.clip(tail_quantile, 0.0, 1.0)
        self.mad_multiplier = max(mad_multiplier, 0.0)
        self.max_ratio = max_ratio if (max_ratio is None or max_ratio > 0) else None
        self.patch_filter_ratio = max(patch_filter_ratio, 0.0)
        self.patch_filter_subsample = patch_filter_subsample
        self.patch_filter_dynamic = patch_filter_dynamic
        self.patch_filter_mad_multiplier = max(patch_filter_mad_multiplier, 0.0)
        self.dynamic_gap = max(float(dynamic_gap), 0.0)

    @staticmethod
    def _robust_covariance(features: np.ndarray, shrinkage: float) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate mean and (shrinked) covariance for the given features."""

        mean = np.mean(features, axis=0)
        centered = features - mean
        cov = np.cov(centered, rowvar=False)
        if np.isscalar(cov):
            cov = np.array([[cov]])
        eye = np.eye(cov.shape[0], dtype=np.float32)
        trace = np.trace(cov)
        if not np.isfinite(trace) or trace <= 0:
            trace = np.mean(np.diag(cov)) * cov.shape[0]
        shrinked = (1 - shrinkage) * cov + shrinkage * (trace / cov.shape[0]) * eye
        return mean, shrinked

    @staticmethod
    def _mahalanobis(centered: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute squared Mahalanobis distances with stability guards."""

        cov = cov.copy()
        cov += 1e-6 * np.eye(cov.shape[0], dtype=cov.dtype)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        return np.einsum("...i,ij,...j->...", centered, inv_cov, centered)

    @staticmethod
    def _quantile(values: np.ndarray, q: float) -> float:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return float("nan")
        q = float(np.clip(q, 0.0, 1.0))
        try:
            return float(np.quantile(values, q, method="linear"))
        except TypeError:
            return float(np.quantile(values, q, interpolation="linear"))

    def mine(
        self,
        per_image_features: Sequence[np.ndarray],
        forced_anomaly_indices: Optional[Sequence[int]] = None,
    ) -> MiningResult:
        normalized_features: List[np.ndarray] = []
        pooled_features = []
        patch_samples = []
        forced_set = set()
        if forced_anomaly_indices is not None:
            for idx in forced_anomaly_indices:
                try:
                    candidate = int(idx)
                except (TypeError, ValueError):
                    continue
                if candidate < 0:
                    continue
                forced_set.add(candidate)
        for feats in per_image_features:
            feats = np.asarray(feats, dtype=np.float32)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            normalized_features.append(feats)
            pooled_features.append(np.mean(feats, axis=0))
            if self.patch_filter_ratio > 0:
                if (
                    self.patch_filter_subsample is not None
                    and feats.shape[0] > self.patch_filter_subsample
                ):
                    choice = self.random_state.choice(
                        feats.shape[0], size=self.patch_filter_subsample, replace=False
                    )
                    patch_samples.append(feats[choice])
                else:
                    patch_samples.append(feats)

        if not normalized_features or self.ratio <= 0:
            return MiningResult(
                normalized_features,
                None,
                [],
                np.zeros(len(normalized_features), dtype=np.float32),
            )

        if not pooled_features:
            LOGGER.warning(
                "Pseudo anomaly miner received only empty feature tensors; skipping mining."
            )
            return MiningResult(
                normalized_features,
                None,
                [],
                np.zeros(len(normalized_features), dtype=np.float32),
            )

        pooled_features = np.stack(pooled_features, axis=0)

        mean, cov = self._robust_covariance(pooled_features, self.shrinkage)
        centered = pooled_features - mean
        scores = self._mahalanobis(centered, cov)

        patch_threshold = None
        patch_mean = None
        patch_cov = None
        if patch_samples and self.patch_filter_ratio > 0:
            patch_concat = np.concatenate(patch_samples, axis=0)
            if patch_concat.shape[0] > 500_000:
                choice = self.random_state.choice(
                    patch_concat.shape[0], size=500_000, replace=False
                )
                patch_concat = patch_concat[choice]
            patch_mean, patch_cov = self._robust_covariance(patch_concat, self.shrinkage)
            patch_centered = patch_concat - patch_mean
            patch_scores = self._mahalanobis(patch_centered, patch_cov)
            quantile_cutoff = self._quantile(patch_scores, 1.0 - self.patch_filter_ratio)
            patch_threshold = quantile_cutoff
            if self.patch_filter_dynamic:
                mu_scores = np.median(patch_scores)
                mad_scores = np.median(np.abs(patch_scores - mu_scores)) + 1e-6
                dynamic_cutoff = mu_scores + self.patch_filter_mad_multiplier * mad_scores
                patch_threshold = max(quantile_cutoff, dynamic_cutoff)

        num_candidates = max(int(len(normalized_features) * self.ratio), self.min_count)
        num_candidates = min(num_candidates, len(normalized_features))

        candidate_indices: np.ndarray
        candidate_indices = np.argsort(scores)[-num_candidates:]

        selection_threshold = None
        if self.dynamic_ratio and len(normalized_features) >= self.min_count:
            mu_scores = np.median(scores)
            mad_scores = np.median(np.abs(scores - mu_scores)) + 1e-6
            mad_cutoff = mu_scores + self.mad_multiplier * mad_scores
            quantile_cutoff = self._quantile(scores, self.tail_quantile)
            selection_threshold = max(mad_cutoff, quantile_cutoff)
            dynamic_indices = np.flatnonzero(scores >= selection_threshold)
            candidate_indices = np.union1d(candidate_indices, dynamic_indices)

        if candidate_indices.size < self.min_count:
            top_required = np.argsort(scores)[-self.min_count :]
            candidate_indices = np.union1d(candidate_indices, top_required)

        if self.max_ratio is not None:
            max_candidates = max(int(len(normalized_features) * self.max_ratio), self.min_count)
            if candidate_indices.size > max_candidates:
                top_sorted = np.argsort(scores)[-max_candidates:]
                candidate_indices = np.union1d(candidate_indices, top_sorted)
                # Union may exceed max_candidates; enforce by selecting highest scores.
                if candidate_indices.size > max_candidates:
                    ordered = candidate_indices[np.argsort(scores[candidate_indices])]
                    candidate_indices = ordered[-max_candidates:]

        top_indices = np.sort(candidate_indices.astype(int))
        separation_gap: Optional[float] = None
        selected_ratio: Optional[float] = None

        if top_indices.size:
            selected_ratio = top_indices.size / max(len(normalized_features), 1)
            if top_indices.size < len(normalized_features):
                mask = np.ones(len(scores), dtype=bool)
                mask[top_indices] = False
                pseudo_scores = scores[top_indices]
                normal_scores_subset = (
                    scores[mask] if np.any(mask) else np.array([], dtype=scores.dtype)
                )
                if pseudo_scores.size and normal_scores_subset.size:
                    separation_gap = float(
                        np.median(pseudo_scores) - np.median(normal_scores_subset)
                    )
                elif pseudo_scores.size:
                    separation_gap = float(np.median(pseudo_scores))

        if (
            self.dynamic_ratio
            and self.dynamic_gap > 0
            and top_indices.size > 0
            and top_indices.size < len(normalized_features)
        ):
            max_candidates = len(normalized_features)
            if self.max_ratio is not None:
                max_candidates = max(int(len(normalized_features) * self.max_ratio), self.min_count)
            max_candidates = min(max_candidates, len(normalized_features))
            current = top_indices.size
            if separation_gap is None:
                separation_gap = float(np.median(scores[top_indices]) - np.median(scores))
            while (
                separation_gap is not None
                and separation_gap < self.dynamic_gap
                and current < max_candidates
            ):
                new_current = min(
                    max_candidates,
                    max(current + 1, int(np.ceil(current * 1.5))),
                )
                if new_current <= current:
                    break
                candidate_indices = np.argsort(scores)[-new_current:]
                top_indices = np.sort(candidate_indices.astype(int))
                current = new_current
                mask = np.ones(len(scores), dtype=bool)
                mask[top_indices] = False
                pseudo_scores = scores[top_indices]
                normal_scores_subset = (
                    scores[mask] if np.any(mask) else np.array([], dtype=scores.dtype)
                )
                if pseudo_scores.size and normal_scores_subset.size:
                    separation_gap = float(np.median(pseudo_scores) - np.median(normal_scores_subset))
                elif pseudo_scores.size:
                    separation_gap = float(np.median(pseudo_scores))
                else:
                    separation_gap = None

            selected_ratio = top_indices.size / max(len(normalized_features), 1)

        if separation_gap is not None and not np.isfinite(separation_gap):
            separation_gap = None

        if forced_set:
            forced_indices = np.fromiter(
                (idx for idx in forced_set if idx < len(per_image_features)), dtype=int
            )
            if forced_indices.size:
                top_indices = np.union1d(top_indices, forced_indices)

        if top_indices.size:
            mask = np.ones(len(scores), dtype=bool)
            mask[top_indices] = False
            pseudo_scores = scores[top_indices]
            normal_scores_subset = (
                scores[mask] if np.any(mask) else np.array([], dtype=scores.dtype)
            )
            selected_ratio = top_indices.size / max(len(normalized_features), 1)
            if pseudo_scores.size and normal_scores_subset.size:
                separation_gap = float(
                    np.median(pseudo_scores) - np.median(normal_scores_subset)
                )
            elif pseudo_scores.size:
                separation_gap = float(np.median(pseudo_scores))
        else:
            selected_ratio = 0.0

        LOGGER.info(
            "Pseudo anomaly miner selected %d / %d training samples as hard negatives (%.2f%%).",
            int(top_indices.size),
            len(per_image_features),
            100.0 * selected_ratio if selected_ratio is not None else 0.0,
        )
        if separation_gap is not None:
            LOGGER.info(
                "Pseudo anomaly median gap (pseudo - normal): %.4f",
                separation_gap,
            )

        memory_features: List[np.ndarray] = []
        pseudo_features: List[np.ndarray] = []
        normal_calibration_features: List[np.ndarray] = []
        pseudo_calibration_features: List[np.ndarray] = []
        rejected_patches = 0
        kept_patches = 0
        pseudo_index_set = set(top_indices.tolist()) if top_indices.size else set()
        for idx, feats in enumerate(normalized_features):
            feats = np.asarray(feats, dtype=np.float32)
            if feats.size == 0:
                continue

            filtered = feats
            kept = feats.shape[0]
            rejected = 0
            if (
                patch_threshold is not None
                and patch_mean is not None
                and patch_cov is not None
                and feats.size > 0
            ):
                patch_centered = feats - patch_mean
                patch_scores = self._mahalanobis(patch_centered, patch_cov)
                keep_mask = patch_scores <= patch_threshold
                kept = int(np.sum(keep_mask))
                rejected = int(keep_mask.size - kept)
                if kept == 0:
                    keep_index = int(np.argmin(patch_scores))
                    filtered = feats[[keep_index]]
                    kept = 1
                    rejected = max(int(patch_scores.size) - 1, 0)
                else:
                    filtered = feats[keep_mask]

            kept_patches += kept
            rejected_patches += rejected

            if filtered.size > 0:
                memory_features.append(filtered)

            if idx in pseudo_index_set:
                pseudo_calibration_features.append(feats)
                if self.patch_subsample is not None and feats.shape[0] > self.patch_subsample:
                    choice = self.random_state.choice(
                        feats.shape[0], size=self.patch_subsample, replace=False
                    )
                    pseudo_features.append(feats[choice])
                else:
                    pseudo_features.append(feats)
            else:
                normal_calibration_features.append(filtered)

        pseudo_array = None
        if pseudo_features:
            pseudo_array = np.concatenate(pseudo_features, axis=0)

        if patch_threshold is not None:
            LOGGER.info(
                "Patch-level filter kept %d patches and rejected %d (threshold=%.4f)",
                kept_patches,
                rejected_patches,
                patch_threshold,
            )
        else:
            kept_patches = int(sum(feats.shape[0] for feats in memory_features))
            rejected_patches = 0

        return MiningResult(
            memory_features,
            pseudo_array,
            top_indices,
            scores.astype(np.float32),
            selection_threshold,
            patch_threshold,
            rejected_patches,
            kept_patches,
            normal_calibration_features if normal_calibration_features else None,
            pseudo_calibration_features if pseudo_calibration_features else None,
            separation_gap=separation_gap,
            selected_ratio=selected_ratio,
        )


class ScoreCalibrator:
    """Map raw kNN distances to calibrated anomaly probabilities."""

    def __init__(
        self,
        temperature: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        target_fpr: float = 0.05,
        target_fnr: float = 0.05,
        strategy: str = "balanced",
        max_search_grid: int = 4096,
    ) -> None:
        self.temperature = max(temperature, 1e-6)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.shift = 0.0
        self.scale = 1.0
        self.fitted = False
        self.target_fpr = max(min(target_fpr, 1.0), 0.0)
        self.target_fnr = max(min(target_fnr, 1.0), 0.0)
        self.raw_threshold: Optional[float] = None
        self.calibrated_threshold: Optional[float] = None
        self.normal_reference: Optional[np.ndarray] = None
        self.anomaly_reference: Optional[np.ndarray] = None
        strategy = strategy.lower()
        if strategy not in {"grid", "balanced", "quantile"}:
            raise ValueError(f"Unsupported calibration strategy: {strategy}")
        self.strategy = strategy
        self.max_search_grid = max(int(max_search_grid), 32)
        self.calibration_fpr: Optional[float] = None
        self.calibration_fnr: Optional[float] = None
        self.last_fit_stats: Optional[dict] = None

    @staticmethod
    def _to_numpy(values: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float32)
        if array.ndim == 0:
            array = array.reshape(1)
        return array

    @staticmethod
    def _prepare_weights(
        values: Sequence[float], weights: Optional[Sequence[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        array = ScoreCalibrator._to_numpy(values)
        if weights is None:
            weight_array = np.ones_like(array, dtype=np.float32)
        else:
            weight_array = ScoreCalibrator._to_numpy(weights)
            if weight_array.shape != array.shape:
                raise ValueError("weights must match value shape")
            weight_array = np.maximum(weight_array.astype(np.float32), 0.0)
        total = float(np.sum(weight_array))
        if not np.isfinite(total) or total <= 0:
            weight_array = np.ones_like(array, dtype=np.float32)
            total = float(np.sum(weight_array))
        weight_array = weight_array / total
        return array, weight_array

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray, weights: np.ndarray, quantile: float
    ) -> float:
        quantile = float(np.clip(quantile, 0.0, 1.0))
        if values.size == 0:
            return float("nan")
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
        cumulative = np.cumsum(weights)
        if cumulative[-1] <= 0:
            return float(values[-1])
        cumulative = cumulative / cumulative[-1]
        return float(np.interp(quantile, cumulative, values))

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        return ScoreCalibrator._weighted_quantile(values, weights, 0.5)

    @staticmethod
    def _weighted_mad(values: np.ndarray, weights: np.ndarray, median: float) -> float:
        deviations = np.abs(values - median)
        return ScoreCalibrator._weighted_quantile(deviations, weights, 0.5)

    @staticmethod
    def _evaluate_threshold(
        threshold: float,
        normal_scores: np.ndarray,
        normal_weights: np.ndarray,
        anomaly_scores: Optional[np.ndarray],
        anomaly_weights: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        false_alarm = float(
            np.sum(normal_weights[normal_scores > threshold])
        )
        total_normal = float(np.sum(normal_weights))
        fpr = false_alarm / max(total_normal, 1e-6)

        if anomaly_scores is None or anomaly_scores.size == 0:
            return fpr, float("nan")

        if anomaly_weights is None:
            anomaly_weights = np.ones_like(anomaly_scores, dtype=np.float32)

        miss = float(
            np.sum(anomaly_weights[anomaly_scores <= threshold])
        )
        total_anomaly = float(np.sum(anomaly_weights))
        fnr = miss / max(total_anomaly, 1e-6)
        return fpr, fnr

    def fit(
        self,
        normal_scores: Sequence[float],
        anomaly_scores: Optional[Sequence[float]] = None,
        normal_weights: Optional[Sequence[float]] = None,
        anomaly_weights: Optional[Sequence[float]] = None,
    ) -> None:
        normal_scores, normal_weights_array = self._prepare_weights(
            normal_scores, normal_weights
        )
        if normal_scores.size == 0:
            raise ValueError("normal_scores must not be empty")

        mu_normal = self._weighted_median(normal_scores, normal_weights_array)
        mad = self._weighted_mad(normal_scores, normal_weights_array, mu_normal) + 1e-6

        anomaly_scores_array = None
        anomaly_weights_array = None
        if anomaly_scores is not None:
            anomaly_scores_array, anomaly_weights_array = self._prepare_weights(
                anomaly_scores, anomaly_weights
            )
            if anomaly_scores_array.size > 0:
                mu_anomaly = self._weighted_median(
                    anomaly_scores_array, anomaly_weights_array
                )
            else:
                mu_anomaly = mu_normal + mad
        else:
            mu_anomaly = mu_normal + mad

        self.shift = 0.5 * (mu_normal + mu_anomaly)
        spread = abs(mu_anomaly - mu_normal)
        if spread < 1e-6:
            spread = mad
        self.scale = max(spread / self.temperature, 1e-6)
        self.fitted = True

        self.normal_reference = normal_scores
        self.anomaly_reference = anomaly_scores_array
        raw_threshold, cal_fpr, cal_fnr = self._compute_raw_threshold(
            normal_scores,
            anomaly_scores_array,
            normal_weights_array,
            anomaly_weights_array,
        )
        self.raw_threshold = raw_threshold
        if self.raw_threshold is not None:
            self.calibrated_threshold = self([self.raw_threshold])[0]
        else:
            self.calibrated_threshold = None
        self.calibration_fpr = cal_fpr
        self.calibration_fnr = cal_fnr

        LOGGER.info(
            "Score calibrator fitted with μ_normal=%.4f, μ_anomaly=%.4f, temperature=%.3f",
            mu_normal,
            mu_anomaly,
            self.temperature,
        )

        if self.raw_threshold is not None:
            cal_fpr_pct = 100.0 * cal_fpr if cal_fpr is not None else float("nan")
            if cal_fnr is None or not np.isfinite(cal_fnr):
                cal_fnr_display = "N/A"
            else:
                cal_fnr_display = f"{100.0 * cal_fnr:.2f}%"
            LOGGER.info(
                "Calibrated decision threshold (strategy=%s, target FPR %.2f%%, FNR %.2f%%): raw=%.4f, prob=%.4f, cal FPR=%.2f%%, cal FNR=%s",
                self.strategy,
                100 * self.target_fpr,
                100 * self.target_fnr,
                self.raw_threshold,
                self.calibrated_threshold if self.calibrated_threshold is not None else float("nan"),
                cal_fpr_pct,
                cal_fnr_display,
            )

        self.last_fit_stats = {
            "strategy": self.strategy,
            "target_fpr": float(self.target_fpr),
            "target_fnr": float(self.target_fnr),
            "raw_threshold": float(self.raw_threshold)
            if self.raw_threshold is not None
            else None,
            "prob_threshold": float(self.calibrated_threshold)
            if self.calibrated_threshold is not None
            else None,
            "calibration_fpr": float(cal_fpr) if cal_fpr is not None else None,
            "calibration_fnr": float(cal_fnr) if cal_fnr is not None else None,
        }

    def _compute_raw_threshold(
        self,
        normal_scores: np.ndarray,
        anomaly_scores: Optional[np.ndarray],
        normal_weights: np.ndarray,
        anomaly_weights: Optional[np.ndarray],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if normal_scores.size == 0:
            return None, None, None

        high_normal = self._weighted_quantile(
            normal_scores, normal_weights, 1.0 - self.target_fpr
        )
        candidate_thresholds = np.unique(normal_scores.flatten())
        if anomaly_scores is not None and anomaly_scores.size > 0:
            candidate_thresholds = np.union1d(
                candidate_thresholds, anomaly_scores.flatten()
            )

        candidate_thresholds = np.union1d(candidate_thresholds, [high_normal])

        low_anomaly = None
        if anomaly_scores is not None and anomaly_scores.size > 0:
            if anomaly_weights is None:
                anomaly_weights = np.ones_like(anomaly_scores, dtype=np.float32)
            low_anomaly = self._weighted_quantile(
                anomaly_scores, anomaly_weights, self.target_fnr
            )
            candidate_thresholds = np.union1d(candidate_thresholds, [low_anomaly])

        if candidate_thresholds.size > self.max_search_grid:
            quantiles = np.linspace(0.0, 1.0, num=self.max_search_grid)
            try:
                candidate_thresholds = np.unique(
                    np.quantile(candidate_thresholds, quantiles, method="linear")
                )
            except TypeError:  # pragma: no cover - NumPy compatibility
                candidate_thresholds = np.unique(
                    np.quantile(
                        candidate_thresholds, quantiles, interpolation="linear"
                    )
                )

        if self.strategy == "quantile":
            threshold = high_normal
            if low_anomaly is not None and high_normal > low_anomaly:
                threshold = 0.5 * (high_normal + low_anomaly)
            fpr, fnr = self._evaluate_threshold(
                threshold,
                normal_scores,
                normal_weights,
                anomaly_scores,
                anomaly_weights,
            )
            return float(threshold), fpr, fnr

        best_threshold = float(candidate_thresholds[-1])
        best_fpr, best_fnr = self._evaluate_threshold(
            best_threshold,
            normal_scores,
            normal_weights,
            anomaly_scores,
            anomaly_weights,
        )
        best_cost = float("inf")

        for threshold in candidate_thresholds:
            fpr, fnr = self._evaluate_threshold(
                threshold,
                normal_scores,
                normal_weights,
                anomaly_scores,
                anomaly_weights,
            )
            meets_fpr = fpr <= self.target_fpr
            meets_fnr = np.isnan(fnr) or fnr <= self.target_fnr
            if meets_fpr and meets_fnr:
                return float(threshold), fpr, fnr

            if self.strategy == "balanced":
                fpr_excess = max(fpr - self.target_fpr, 0.0) / max(self.target_fpr, 1e-6)
                if np.isnan(fnr):
                    fnr_excess = 0.0
                else:
                    fnr_excess = max(fnr - self.target_fnr, 0.0) / max(
                        self.target_fnr, 1e-6
                    )
                cost = max(fpr_excess, fnr_excess)
            else:
                cost = max(fpr - self.target_fpr, 0.0) ** 2
                if not np.isnan(fnr):
                    cost += max(fnr - self.target_fnr, 0.0) ** 2
                cost += 1e-6 * (fpr + (fnr if not np.isnan(fnr) else 0.0))

            if cost < best_cost:
                best_cost = cost
                best_threshold = float(threshold)
                best_fpr, best_fnr = fpr, fnr

        return best_threshold, best_fpr, best_fnr

    def __call__(self, raw_scores: Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ScoreCalibrator must be fitted before calling.")
        scores = self._to_numpy(raw_scores)
        logits = (scores - self.shift) / self.scale
        logits = np.clip(logits, -60.0, 60.0)
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(calibrated, self.clip_min, self.clip_max)

    def decision(self, raw_scores: Sequence[float]) -> np.ndarray:
        """Return binary anomaly decisions based on the calibrated threshold."""

        if self.calibrated_threshold is None:
            raise RuntimeError("Calibrated threshold unavailable; call fit() first.")
        probabilities = self(raw_scores)
        return (probabilities >= self.calibrated_threshold).astype(np.uint8)


class PositiveUnlabeledWeighter:
    """Estimate contamination-aware sample weights for calibration.

    The weighting strategy is inspired by the expectation-maximisation routines
    used in "Rank & Sort: Score-Based Calibration for Industrial Anomaly
    Detection" (CVPR 2023) and the positive-unlabeled refinement in "CleanAD:
    Pseudo-Label Guided Self-Ensemble for Contaminated Industrial Anomaly
    Detection" (CVPR 2024). We treat mined pseudo anomalies as noisy positives
    and re-estimate per-sample normal/anomalous responsibilities so that the
    subsequent calibrator can focus on the clean core of the normal set while
    still preserving a small miss rate on pseudo anomalies.
    """

    def __init__(
        self,
        max_iterations: int = 25,
        tolerance: float = 1e-4,
        prior_floor: float = 0.01,
        prior_ceiling: float = 0.5,
        pseudo_confidence_floor: float = 0.7,
        clean_tail_ratio: float = 0.5,
        min_weight: float = 1e-3,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(float(tolerance), 0.0)
        self.prior_floor = float(np.clip(prior_floor, 0.0, 1.0))
        self.prior_ceiling = float(np.clip(prior_ceiling, self.prior_floor, 1.0))
        self.pseudo_confidence_floor = float(np.clip(pseudo_confidence_floor, 0.0, 1.0))
        self.clean_tail_ratio = float(np.clip(clean_tail_ratio, 0.0, 1.0))
        self.min_weight = max(float(min_weight), 1e-6)

    @staticmethod
    def _to_numpy(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 0:
            array = array.reshape(1)
        return array

    @staticmethod
    def _normal_pdf(values: np.ndarray, mean: float, var: float) -> np.ndarray:
        var = max(var, 1e-6)
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = -0.5 * ((values - mean) ** 2) / var
        return coeff * np.exp(exponent)

    def compute_weights(
        self,
        normal_scores: Sequence[float],
        pseudo_scores: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        normal_scores = self._to_numpy(normal_scores)
        pseudo_scores = self._to_numpy(pseudo_scores)

        normal_weights = np.ones_like(normal_scores, dtype=np.float32)
        pseudo_weights = np.ones_like(pseudo_scores, dtype=np.float32)
        diagnostics = {}

        if normal_scores.size == 0 or pseudo_scores.size == 0:
            diagnostics["prior"] = 0.0
            diagnostics["normal_clean_ratio"] = 1.0
            diagnostics["pseudo_confidence"] = 1.0 if pseudo_scores.size else 0.0
            return normal_weights, pseudo_weights, diagnostics

        sorted_indices = np.argsort(normal_scores)
        clean_count = max(int(normal_scores.size * self.clean_tail_ratio), 1)
        clean_subset = normal_scores[sorted_indices[:clean_count]]
        mu_normal = float(np.mean(clean_subset))
        var_normal = float(np.var(clean_subset) + 1e-6)

        mu_anomaly = float(np.mean(pseudo_scores))
        var_anomaly = float(np.var(pseudo_scores) + 1e-6)

        total_count = float(normal_scores.size + pseudo_scores.size)
        prior = np.clip(float(pseudo_scores.size / total_count), self.prior_floor, self.prior_ceiling)

        previous_mu_normal = mu_normal
        previous_mu_anomaly = mu_anomaly
        previous_prior = prior

        for _ in range(self.max_iterations):
            normal_pdf_normal = self._normal_pdf(normal_scores, mu_normal, var_normal)
            normal_pdf_anomaly = self._normal_pdf(normal_scores, mu_anomaly, var_anomaly)
            denom_normal = prior * normal_pdf_anomaly + (1.0 - prior) * normal_pdf_normal + 1e-12
            anomaly_resp = np.clip(prior * normal_pdf_anomaly / denom_normal, 0.0, 1.0)

            pseudo_pdf_normal = self._normal_pdf(pseudo_scores, mu_normal, var_normal)
            pseudo_pdf_anomaly = self._normal_pdf(pseudo_scores, mu_anomaly, var_anomaly)
            denom_pseudo = prior * pseudo_pdf_anomaly + (1.0 - prior) * pseudo_pdf_normal + 1e-12
            pseudo_resp = np.clip(
                prior * pseudo_pdf_anomaly / denom_pseudo,
                self.pseudo_confidence_floor,
                1.0,
            )

            anomaly_weight_sum = float(np.sum(anomaly_resp) + np.sum(pseudo_resp))
            normal_weight_sum = float(np.sum(1.0 - anomaly_resp))

            if anomaly_weight_sum <= 0:
                break

            mu_anomaly = float(
                (
                    np.sum(pseudo_resp * pseudo_scores)
                    + np.sum(anomaly_resp * normal_scores)
                )
                / anomaly_weight_sum
            )
            var_anomaly = float(
                (
                    np.sum(pseudo_resp * (pseudo_scores - mu_anomaly) ** 2)
                    + np.sum(anomaly_resp * (normal_scores - mu_anomaly) ** 2)
                )
                / anomaly_weight_sum
            )
            var_anomaly = max(var_anomaly, 1e-6)

            if normal_weight_sum > 0:
                mu_normal = float(
                    np.sum((1.0 - anomaly_resp) * normal_scores) / normal_weight_sum
                )
                var_normal = float(
                    np.sum((1.0 - anomaly_resp) * (normal_scores - mu_normal) ** 2)
                    / normal_weight_sum
                )
                var_normal = max(var_normal, 1e-6)

            prior = np.clip(
                anomaly_weight_sum / total_count,
                self.prior_floor,
                self.prior_ceiling,
            )

            delta = max(
                abs(mu_normal - previous_mu_normal),
                abs(mu_anomaly - previous_mu_anomaly),
                abs(prior - previous_prior),
            )
            previous_mu_normal = mu_normal
            previous_mu_anomaly = mu_anomaly
            previous_prior = prior
            if delta < self.tolerance:
                break

        normal_weights = np.clip(1.0 - anomaly_resp, self.min_weight, 1.0)
        pseudo_weights = np.clip(pseudo_resp, self.min_weight, 1.0)

        diagnostics["prior"] = float(prior)
        diagnostics["normal_clean_ratio"] = float(
            np.sum(normal_weights > 0.5) / max(normal_weights.size, 1)
        )
        diagnostics["pseudo_confidence"] = float(np.mean(pseudo_resp))
        diagnostics["mu_normal"] = float(mu_normal)
        diagnostics["mu_anomaly"] = float(mu_anomaly)

        return normal_weights, pseudo_weights, diagnostics


class PositiveUnlabeledLogisticCalibrator:
    """Logistic fusion head for contaminated calibration features.

    Recent limited-supervision anomaly detectors such as "Dual Student: Breaking
    the Limits of Few-shot Anomaly Detection" (CVPR 2022), "Rank & Sort: Score
    Based Calibration for Industrial Anomaly Detection" (CVPR 2023), and
    "CleanAD: Pseudo-Label Guided Self-Ensemble for Contaminated Industrial
    Anomaly Detection" (CVPR 2024) highlight the benefit of jointly exploiting
    mined pseudo anomalies and clean normals. Inspired by these works we train a
    lightweight positive-unlabeled logistic regressor on aggregated image-level
    descriptors. The regressor respects the estimated contamination weights and
    targets explicit false-alarm/miss-rate constraints, yielding a calibrated
    probability that can be blended with classical PatchCore scores.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iterations: int = 200,
        l2_penalty: float = 1e-3,
        target_fpr: float = 0.05,
        target_fnr: float = 0.05,
        tolerance: float = 1e-5,
    ) -> None:
        self.learning_rate = max(float(learning_rate), 1e-6)
        self.max_iterations = max(int(max_iterations), 1)
        self.l2_penalty = max(float(l2_penalty), 0.0)
        self.target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
        self.target_fnr = float(np.clip(target_fnr, 0.0, 1.0))
        self.tolerance = max(float(tolerance), 0.0)

        self.weight_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.threshold_: Optional[float] = None
        self.fitted: bool = False
        self.last_fit_stats: Optional[Dict[str, float]] = None

    @staticmethod
    def _prepare_arrays(
        features: Sequence[Sequence[float]], weights: Optional[Sequence[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Features must be a 2D array with shape (N, D).")
        if matrix.shape[0] == 0:
            raise ValueError("Feature matrix must not be empty.")
        if weights is None:
            weight_array = np.ones(matrix.shape[0], dtype=np.float32)
        else:
            weight_array = np.asarray(weights, dtype=np.float32)
            if weight_array.shape[0] != matrix.shape[0]:
                raise ValueError("Weight array must match number of samples.")
            weight_array = np.clip(weight_array, 0.0, None)
        total = float(np.sum(weight_array))
        if not np.isfinite(total) or total <= 0:
            weight_array = np.ones(matrix.shape[0], dtype=np.float32)
        else:
            weight_array = weight_array.astype(np.float32)
        return matrix.astype(np.float32), weight_array.astype(np.float32)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _weighted_threshold(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Tuple[float, float, float]:
        unique_probs = np.unique(probabilities)
        if unique_probs.size > 2048:
            try:
                unique_probs = np.unique(
                    np.quantile(unique_probs, np.linspace(0.0, 1.0, 2048), method="linear")
                )
            except TypeError:
                unique_probs = np.unique(
                    np.quantile(
                        unique_probs, np.linspace(0.0, 1.0, 2048), interpolation="linear"
                    )
                )

        normal_mask = labels <= 0.5
        anomaly_mask = ~normal_mask
        normal_weight = float(np.sum(sample_weights[normal_mask]))
        anomaly_weight = float(np.sum(sample_weights[anomaly_mask]))

        best_cost = float("inf")
        best_threshold = float(unique_probs[-1])
        best_fpr = 1.0
        best_fnr = 1.0

        for threshold in unique_probs:
            pred_anomaly = probabilities >= threshold
            false_alarm = float(np.sum(sample_weights[pred_anomaly & normal_mask]))
            miss = float(np.sum(sample_weights[(~pred_anomaly) & anomaly_mask]))
            fpr = false_alarm / max(normal_weight, 1e-6)
            fnr = miss / max(anomaly_weight, 1e-6)
            meets = fpr <= self.target_fpr and fnr <= self.target_fnr
            cost = max(fpr - self.target_fpr, 0.0) ** 2 + max(
                fnr - self.target_fnr, 0.0
            ) ** 2
            cost += 1e-6 * (fpr + fnr)
            if meets:
                return float(threshold), float(fpr), float(fnr)
            if cost < best_cost:
                best_cost = cost
                best_threshold = float(threshold)
                best_fpr = float(fpr)
                best_fnr = float(fnr)

        return best_threshold, best_fpr, best_fnr

    def fit(
        self,
        normal_features: Sequence[Sequence[float]],
        pseudo_features: Sequence[Sequence[float]],
        normal_weights: Optional[Sequence[float]] = None,
        pseudo_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        normal_matrix, normal_weight_array = self._prepare_arrays(
            normal_features, normal_weights
        )
        pseudo_matrix, pseudo_weight_array = self._prepare_arrays(
            pseudo_features, pseudo_weights
        )

        if normal_matrix.shape[1] != pseudo_matrix.shape[1]:
            raise ValueError("Feature dimensionality mismatch between classes.")

        feature_dim = int(normal_matrix.shape[1])
        combined = np.concatenate([normal_matrix, pseudo_matrix], axis=0)
        labels = np.concatenate(
            [
                np.zeros(normal_matrix.shape[0], dtype=np.float32),
                np.ones(pseudo_matrix.shape[0], dtype=np.float32),
            ]
        )
        weights = np.concatenate([normal_weight_array, pseudo_weight_array], axis=0)
        weights = np.asarray(weights, dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            weights = np.ones_like(weights, dtype=np.float32)
            weight_sum = float(weights.size)
        else:
            weights = weights * (weights.size / weight_sum)

        augmented = np.concatenate([combined, np.ones((combined.shape[0], 1), dtype=np.float32)], axis=1)
        params = np.zeros(feature_dim + 1, dtype=np.float32)

        for iteration in range(self.max_iterations):
            logits = augmented @ params
            probs = self._sigmoid(logits)
            diff = probs - labels
            gradient = augmented.T @ (weights * diff)
            gradient[:-1] += self.l2_penalty * params[:-1]

            step = self.learning_rate / np.sqrt(iteration + 1.0)
            params -= step * gradient.astype(np.float32)

            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm < self.tolerance:
                break

        logits = augmented @ params
        probabilities = self._sigmoid(logits)

        threshold, cal_fpr, cal_fnr = self._weighted_threshold(
            probabilities, labels, weights
        )

        self.weight_ = params[:-1].astype(np.float32)
        self.bias_ = float(params[-1])
        self.threshold_ = float(threshold)
        self.fitted = True

        normal_prob = probabilities[: normal_matrix.shape[0]]
        pseudo_prob = probabilities[normal_matrix.shape[0] :]

        normal_weight_sum = float(np.sum(weights[: normal_matrix.shape[0]]))
        pseudo_weight_sum = float(np.sum(weights[normal_matrix.shape[0] :]))
        prior = pseudo_weight_sum / max(normal_weight_sum + pseudo_weight_sum, 1e-6)
        loss = float(
            -np.sum(
                weights
                * (
                    labels * np.log(probabilities + 1e-12)
                    + (1.0 - labels) * np.log(1.0 - probabilities + 1e-12)
                )
            )
        )

        stats = {
            "prior": float(prior),
            "train_loss": loss,
            "threshold": self.threshold_,
            "calibration_fpr": float(cal_fpr),
            "calibration_fnr": float(cal_fnr),
            "normal_prob_mean": float(np.mean(normal_prob)),
            "pseudo_prob_mean": float(np.mean(pseudo_prob)),
        }
        self.last_fit_stats = stats
        return stats

    def predict_proba(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if not self.fitted or self.weight_ is None:
            raise RuntimeError(
                "PositiveUnlabeledLogisticCalibrator must be fitted before calling predict_proba."
            )
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != int(self.weight_.shape[0]):
            raise ValueError("Unexpected feature dimension for fusion calibrator.")
        logits = matrix @ self.weight_ + self.bias_
        return self._sigmoid(logits)

    def decision(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold undefined; call fit() first.")
        probabilities = self.predict_proba(features)
        return (probabilities >= self.threshold_).astype(np.uint8)

class SemiSupervisedDiscriminator:
    """Linear discriminant head for mined pseudo anomalies.

    This lightweight head borrows ideas from limited anomaly supervision
    techniques such as "RD4AD: Real-world Anomaly Detection via Distribution
    Regularization" (CVPR 2022), "Pseudo Outlier Exposure for Industrial Anomaly
    Detection" (CVPR 2023), and the pseudo-label refinement strategy in
    "CleanAD: Pseudo-Label Guided Self-Ensemble for Contaminated Industrial
    Anomaly Detection" (CVPR 2024). By projecting per-image patch descriptors to
    a pooled representation and learning a shrinkage-regularised Fisher
    discriminant, we can softly incorporate a handful of suspected anomaly
    exemplars into the scoring rule without destabilising the one-class memory
    bank.
    """

    def __init__(
        self,
        pooling: str = "mean-max-std",
        shrinkage: float = 0.05,
        min_normals: int = 16,
        min_pseudo: int = 4,
        temperature: float = 1.0,
    ) -> None:
        pooling = pooling.lower()
        if pooling not in {"mean", "mean-max", "mean-max-std"}:
            raise ValueError("Unsupported pooling strategy: %s" % pooling)
        self.pooling = pooling
        self.shrinkage = max(float(shrinkage), 0.0)
        self.min_normals = max(int(min_normals), 1)
        self.min_pseudo = max(int(min_pseudo), 1)
        self.temperature = max(float(temperature), 1e-6)

        self.weight_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.fitted: bool = False
        self.output_dim_: Optional[int] = None
        self.last_fit_stats: Optional[dict] = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-values))

    def reset(self) -> None:
        self.weight_ = None
        self.bias_ = 0.0
        self.fitted = False
        self.output_dim_ = None
        self.last_fit_stats = None

    def _pool_single(self, features: np.ndarray) -> Optional[np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        if features.size == 0:
            return None
        if features.ndim == 1:
            features = features.reshape(1, -1)
        pooled = [np.mean(features, axis=0)]
        if self.pooling in {"mean-max", "mean-max-std"}:
            pooled.append(np.max(features, axis=0))
        if self.pooling == "mean-max-std":
            pooled.append(np.std(features, axis=0))
        vector = np.concatenate(pooled, axis=0).astype(np.float32)
        return vector

    def transform(self, feature_sets: Sequence[np.ndarray]) -> np.ndarray:
        pooled_vectors: List[np.ndarray] = []
        for feats in feature_sets or []:
            if feats is None:
                continue
            vector = self._pool_single(feats)
            if vector is not None:
                pooled_vectors.append(vector)
        if not pooled_vectors:
            return np.empty((0, 0), dtype=np.float32)
        stacked = np.stack(pooled_vectors, axis=0)
        return stacked.astype(np.float32)

    def fit(
        self,
        normal_feature_sets: Sequence[np.ndarray],
        pseudo_feature_sets: Optional[Sequence[np.ndarray]] = None,
    ) -> dict:
        normal_vectors = self.transform(normal_feature_sets)
        if normal_vectors.size == 0 or normal_vectors.shape[0] < self.min_normals:
            raise ValueError("Insufficient normal samples for semi-supervised head.")

        pseudo_vectors = None
        if pseudo_feature_sets is not None:
            pseudo_vectors = self.transform(pseudo_feature_sets)
        if pseudo_vectors is None or pseudo_vectors.size == 0:
            raise ValueError("No pseudo anomalies available for semi-supervised head.")
        if pseudo_vectors.shape[0] < self.min_pseudo:
            raise ValueError("Insufficient pseudo anomalies for semi-supervised head.")

        self.output_dim_ = int(normal_vectors.shape[1])
        if pseudo_vectors.shape[1] != self.output_dim_:
            raise ValueError("Feature dimension mismatch between classes.")

        normal_weights = np.ones(normal_vectors.shape[0], dtype=np.float32)
        pseudo_weights = np.ones(pseudo_vectors.shape[0], dtype=np.float32)
        normal_weight_sum = float(np.sum(normal_weights))
        pseudo_weight_sum = float(np.sum(pseudo_weights))
        if normal_weight_sum <= 0 or pseudo_weight_sum <= 0:
            raise ValueError("Degenerate weights for semi-supervised head.")

        normal_weights /= normal_weight_sum
        pseudo_weights /= pseudo_weight_sum

        mu_normal = np.average(normal_vectors, axis=0, weights=normal_weights)
        mu_pseudo = np.average(pseudo_vectors, axis=0, weights=pseudo_weights)

        centered_normal = normal_vectors - mu_normal
        centered_pseudo = pseudo_vectors - mu_pseudo
        cov_normal = (normal_weights[:, None] * centered_normal).T @ centered_normal
        cov_pseudo = (pseudo_weights[:, None] * centered_pseudo).T @ centered_pseudo
        pooled_cov = cov_normal + cov_pseudo
        pooled_cov /= max(float(normal_vectors.shape[0] + pseudo_vectors.shape[0]), 1.0)

        trace = float(np.trace(pooled_cov))
        if not np.isfinite(trace) or trace <= 0:
            trace = float(np.mean(np.diag(pooled_cov)) * pooled_cov.shape[0])
        shrink = self.shrinkage
        pooled_cov = (1 - shrink) * pooled_cov + shrink * (trace / pooled_cov.shape[0]) * np.eye(
            pooled_cov.shape[0], dtype=np.float32
        )
        pooled_cov += 1e-6 * np.eye(pooled_cov.shape[0], dtype=np.float32)

        diff = (mu_pseudo - mu_normal).astype(np.float32)
        try:
            weight = np.linalg.solve(pooled_cov, diff)
        except np.linalg.LinAlgError:
            weight = np.linalg.pinv(pooled_cov) @ diff

        prior = np.clip(
            pseudo_weight_sum / (pseudo_weight_sum + normal_weight_sum),
            1e-6,
            1 - 1e-6,
        )

        bias = -0.5 * float((mu_pseudo + mu_normal) @ weight) + float(
            np.log(prior / (1.0 - prior))
        )

        self.weight_ = weight.astype(np.float32)
        self.bias_ = bias
        self.fitted = True

        logits_normal = self.predict_logits(normal_vectors)
        logits_pseudo = self.predict_logits(pseudo_vectors)
        probs_normal = self._sigmoid(logits_normal)
        probs_pseudo = self._sigmoid(logits_pseudo)

        stats = {
            "prior": float(prior),
            "margin": float(np.mean(logits_pseudo) - np.mean(logits_normal)),
            "normal_prob_mean": float(np.mean(probs_normal)),
            "normal_prob_std": float(np.std(probs_normal)),
            "pseudo_prob_mean": float(np.mean(probs_pseudo)),
            "pseudo_prob_std": float(np.std(probs_pseudo)),
            "normal_count": int(normal_vectors.shape[0]),
            "pseudo_count": int(pseudo_vectors.shape[0]),
        }
        self.last_fit_stats = stats
        return stats

    def predict_logits(self, pooled_vectors: Sequence[np.ndarray]) -> np.ndarray:
        if not self.fitted or self.weight_ is None:
            raise RuntimeError("SemiSupervisedDiscriminator must be fitted before use.")
        vectors = np.asarray(pooled_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[-1] != int(self.weight_.shape[0]):
            raise ValueError("Unexpected feature dimension for semi-supervised head.")
        logits = vectors @ self.weight_ + self.bias_
        return logits / self.temperature

    def predict_proba(self, pooled_vectors: Sequence[np.ndarray]) -> np.ndarray:
        logits = self.predict_logits(pooled_vectors)
        return self._sigmoid(logits)

    def pool(self, features: np.ndarray) -> Optional[np.ndarray]:
        vector = self._pool_single(features)
        if vector is None:
            return None
        if self.output_dim_ is None:
            self.output_dim_ = int(vector.shape[0])
        return vector