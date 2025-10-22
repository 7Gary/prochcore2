"""Contamination-aware control modules for advanced PatchCore extensions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ContaminationElasticityState:
    """Container for tracking contamination elasticity statistics."""

    distances: np.ndarray
    elasticity: float
    threshold: float
    contamination_rate: float
    gamma: float
    energy_baseline: float
    pseudo_mask: np.ndarray


class ContaminationElasticityController:
    """Implements the contamination elasticity principle.

    The controller maintains robust statistics of the incoming embeddings using a
    double-median correction, median absolute deviation (MAD) normalisation and
    an exponential smoothing scheme.  It exposes a continuously differentiable
    elasticity value that is monotonic to the estimated contamination ratio and
    provides a dynamic ``gamma`` coefficient for the dual-memory distillation
    process.
    """

    def __init__(
        self,
        feature_dim: int,
        smoothing: float = 0.2,
        elasticity_temperature: float = 2.0,
        minimum_threshold: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        self.feature_dim = feature_dim
        self.smoothing = smoothing
        self.elasticity_temperature = elasticity_temperature
        self.minimum_threshold = minimum_threshold
        self.eps = eps

        self._median: Optional[np.ndarray] = None
        self._offset: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._inv_covariance: Optional[np.ndarray] = None
        self._elasticity: float = 0.0
        self._energy_baseline: float = 0.0
        self._prev_elasticity: float = 0.0
        self._history: List[Dict[str, float]] = []

    @property
    def history(self) -> List[Dict[str, float]]:
        return self._history

    @property
    def elasticity(self) -> float:
        return self._elasticity

    @property
    def gamma(self) -> float:
        # map elasticity to a bounded coefficient in (0, 1)
        return float(1.0 - np.exp(-np.clip(self._elasticity, 0.0, None)))

    def _exponential_smooth(self, previous: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current
        return (1.0 - self.smoothing) * previous + self.smoothing * current

    def _update_statistics(self, embeddings: np.ndarray) -> None:
        median = np.median(embeddings, axis=0)
        self._median = self._exponential_smooth(self._median, median)

        offset = np.median(embeddings - self._median, axis=0)
        self._offset = self._exponential_smooth(self._offset, offset)

        centred = embeddings - self._median - self._offset
        mad = np.median(np.abs(centred), axis=0) + self.eps
        self._mad = self._exponential_smooth(self._mad, mad)

        normalised = centred / self._mad
        covariance = np.cov(normalised, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * self.eps
        self._covariance = self._exponential_smooth(self._covariance, covariance)
        self._inv_covariance = np.linalg.pinv(self._covariance)

    def estimate(self, embeddings: np.ndarray) -> ContaminationElasticityState:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape [N, D].")
        if embeddings.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected embedding dimension {self.feature_dim}, got {embeddings.shape[1]}"
            )
        self._update_statistics(embeddings)

        centred = embeddings - self._median - self._offset
        normalised = centred / self._mad
        distances = np.sqrt(
            np.sum(normalised @ self._inv_covariance * normalised, axis=1)
        )

        median_distance = np.median(distances)
        mad_distance = np.median(np.abs(distances - median_distance)) + self.eps
        z_scores = (distances - median_distance) / mad_distance
        z_scores = np.clip(z_scores, a_min=0.0, a_max=None)

        raw_energy = np.mean(z_scores)
        self._energy_baseline = (
            (1.0 - self.smoothing) * self._energy_baseline + self.smoothing * raw_energy
        )
        contamination_rate = float(np.mean(z_scores > 0))

        # Elasticity curve: smoothed softplus on top of the raw energy term.
        elasticity = np.log1p(np.exp(self.elasticity_temperature * (raw_energy - 1.0)))
        self._elasticity = (1.0 - self.smoothing) * self._elasticity + self.smoothing * elasticity

        elasticity_delta = self._elasticity - self._prev_elasticity
        self._prev_elasticity = self._elasticity

        threshold = float(max(median_distance + mad_distance, self.minimum_threshold))
        pseudo_mask = distances >= threshold

        gamma = float(np.clip(self.gamma + np.tanh(elasticity_delta), 0.0, 1.5))
        state = ContaminationElasticityState(
            distances=distances,
            elasticity=self._elasticity,
            threshold=threshold,
            contamination_rate=contamination_rate,
            gamma=gamma,
            energy_baseline=self._energy_baseline,
            pseudo_mask=pseudo_mask,
        )
        self._history.append(
            {
                "elasticity": state.elasticity,
                "contamination": state.contamination_rate,
                "threshold": state.threshold,
                "gamma": state.gamma,
                "energy": state.energy_baseline,
            }
        )
        return state


class ReciprocalDualMemoryDistillation:
    """Implements reciprocal distillation between normal and pseudo memories."""

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature
        self.energy_trace: List[float] = []
        self.gamma_trace: List[float] = []

    def combine(
        self,
        normal_scores: np.ndarray,
        normal_distances: np.ndarray,
        pseudo_distances: Optional[np.ndarray],
        gamma: float,
        normal_weight: float,
        pseudo_weight: float,
    ) -> np.ndarray:
        forward_energy = np.mean(np.square(normal_distances), axis=-1)

        if pseudo_distances is None:
            response = normal_weight * forward_energy
            self.energy_trace.append(float(np.mean(response)))
            self.gamma_trace.append(float(gamma))
            return response

        backward_energy = np.mean(np.square(pseudo_distances), axis=-1)
        reciprocal_raw = forward_energy - gamma * backward_energy
        reciprocal_raw = np.clip(reciprocal_raw, a_min=0.0, a_max=None)
        suppression = np.exp(-backward_energy / max(self.temperature, 1e-12))
        reciprocal_energy = normal_weight * reciprocal_raw + pseudo_weight * suppression

        self.energy_trace.append(float(np.mean(reciprocal_energy)))
        self.gamma_trace.append(float(gamma))
        return reciprocal_energy


@dataclass
class ClosedLoopState:
    elasticity: float
    control_gain: float
    integral_term: float
    pseudo_weight: float
    normal_weight: float
    temperature: float


class ClosedLoopContaminationControl:
    """Closed-loop controller that stabilises contamination handling."""

    def __init__(
        self,
        elasticity_controller: ContaminationElasticityController,
        alpha: float = 1.2,
        beta: float = 0.5,
        eta: float = 0.1,
        base_temperature: float = 0.05,
        max_temperature: float = 0.5,
    ) -> None:
        self.elasticity_controller = elasticity_controller
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.base_temperature = base_temperature
        self.max_temperature = max_temperature

        self._integral: float = 0.0
        self._previous_elasticity: float = 0.0
        self._state_history: List[ClosedLoopState] = []

    @property
    def history(self) -> List[ClosedLoopState]:
        return self._state_history

    def step(self, contamination_rate: float) -> ClosedLoopState:
        elasticity = self.elasticity_controller.elasticity
        delta = elasticity - self._previous_elasticity
        self._previous_elasticity = elasticity

        self._integral += contamination_rate
        raw_gain = self.alpha * elasticity + self.beta * delta + self.eta * self._integral
        control_gain = float(1.0 / (1.0 + np.exp(-raw_gain)))

        pseudo_weight = np.interp(control_gain, [0.0, 1.0], [0.2, 1.5])
        normal_weight = np.interp(control_gain, [0.0, 1.0], [1.2, 0.6])
        temperature = np.interp(
            control_gain,
            [0.0, 1.0],
            [self.base_temperature, self.max_temperature],
        )

        state = ClosedLoopState(
            elasticity=elasticity,
            control_gain=control_gain,
            integral_term=self._integral,
            pseudo_weight=pseudo_weight,
            normal_weight=normal_weight,
            temperature=temperature,
        )
        self._state_history.append(state)
        return state


class ContaminationDashboard:
    """Collects interpretable metrics for reproducibility and visualisation."""

    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {
            "elasticity": [],
            "contamination": [],
            "threshold": [],
            "gamma": [],
            "energy": [],
            "control_gain": [],
            "pseudo_weight": [],
            "normal_weight": [],
            "temperature": [],
            "dual_energy": [],
            "purity_threshold": [],
            "purity_mean": [],
            "purity_iterations": [],
            "uncertainty_threshold": [],
            "uncertainty_filtered": [],
            "uncertainty_retained": [],
            "diffusion_generated": [],
            "diffusion_ratio": [],
            "scale_forging_loss": [],
        }

    def log_elasticity(self, state: ContaminationElasticityState) -> None:
        self.records["elasticity"].append(float(state.elasticity))
        self.records["contamination"].append(float(state.contamination_rate))
        self.records["threshold"].append(float(state.threshold))
        self.records["gamma"].append(float(state.gamma))
        self.records["energy"].append(float(state.energy_baseline))

    def log_closed_loop(self, state: ClosedLoopState) -> None:
        self.records["control_gain"].append(float(state.control_gain))
        self.records["pseudo_weight"].append(float(state.pseudo_weight))
        self.records["normal_weight"].append(float(state.normal_weight))
        self.records["temperature"].append(float(state.temperature))

    def log_dual_memory(self, energy_trace: List[float]) -> None:
        if energy_trace:
            self.records["dual_energy"].append(float(energy_trace[-1]))

    def log_purity(self, threshold: float, mean_purity: float, iterations: float) -> None:
        self.records["purity_threshold"].append(float(threshold))
        self.records["purity_mean"].append(float(mean_purity))
        self.records["purity_iterations"].append(float(iterations))

    def log_uncertainty(
        self, threshold: float, filtered: float, retained: float
    ) -> None:
        self.records["uncertainty_threshold"].append(float(threshold))
        self.records["uncertainty_filtered"].append(float(filtered))
        self.records["uncertainty_retained"].append(float(retained))

    def log_diffusion(self, generated: int, ratio: float) -> None:
        self.records["diffusion_generated"].append(float(generated))
        self.records["diffusion_ratio"].append(float(ratio))

    def log_dynamic_scale(self, summary: Dict[str, float]) -> None:
        if not summary:
            return
        loss = summary.get("forging_loss")
        if loss is not None:
            self.records["scale_forging_loss"].append(float(loss))

    def export(self) -> Dict[str, List[float]]:
        return {key: list(values) for key, values in self.records.items()}

    def dump(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.export(), handle, indent=2, ensure_ascii=False)