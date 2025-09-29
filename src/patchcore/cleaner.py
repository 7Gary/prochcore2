"""Feature triaging utilities for contamination-aware memory construction.

This module implements a robust trimming heuristic inspired by recent
industrial anomaly detection work such as "Rank & Sort" (CVPR 2023) and
"CleanAD" (CVPR 2024).  We estimate a robust centre using the component-wise
median and median absolute deviation (MAD) of image-level feature summaries and
remove a capped fraction of outliers before forming the PatchCore memory bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class CleanerStats:
    """Diagnostic statistics returned by :class:`RobustFeatureCleaner`."""

    total: int
    removed: int
    threshold: float
    median_distance: float
    mad: float


class RobustFeatureCleaner:
    """Flag suspicious training images before populating the memory bank."""

    def __init__(
        self,
        max_ratio: float = 0.15,
        mad_multiplier: float = 3.5,
        min_keep: int = 8,
    ) -> None:
        if max_ratio is not None and max_ratio < 0:
            raise ValueError("max_ratio must be non-negative or None")
        if mad_multiplier < 0:
            raise ValueError("mad_multiplier must be non-negative")
        if min_keep < 0:
            raise ValueError("min_keep must be non-negative")
        self.max_ratio = max_ratio
        self.mad_multiplier = mad_multiplier
        self.min_keep = min_keep

    def filter(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, CleanerStats]:
        """Return a boolean mask selecting clean training images."""

        total = len(features)
        if total == 0:
            mask = np.zeros(0, dtype=bool)
            stats = CleanerStats(
                total=0,
                removed=0,
                threshold=float("nan"),
                median_distance=float("nan"),
                mad=float("nan"),
            )
            return mask, stats

        pooled = []
        valid_indices = []
        for idx, feats in enumerate(features):
            arr = np.asarray(feats, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            pooled.append(arr.mean(axis=0))
            valid_indices.append(idx)

        mask = np.ones(total, dtype=bool)
        if not pooled:
            stats = CleanerStats(
                total=total,
                removed=0,
                threshold=float("nan"),
                median_distance=float("nan"),
                mad=float("nan"),
            )
            return mask, stats

        pooled_array = np.stack(pooled, axis=0)
        centre = np.median(pooled_array, axis=0)
        diffs = pooled_array - centre
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        median_distance = float(np.median(distances)) if distances.size else float("nan")
        mad = float(1.4826 * np.median(np.abs(distances - median_distance))) if distances.size else float("nan")
        if not np.isfinite(mad) or mad <= 0:
            mad = 0.0
        threshold = float(median_distance + self.mad_multiplier * mad) if np.isfinite(median_distance) else float("nan")

        flagged_pairs = []
        for distance, idx in zip(distances, valid_indices):
            if np.isfinite(threshold) and distance > threshold:
                flagged_pairs.append((float(distance), idx))

        flagged_pairs.sort(reverse=True)
        if self.max_ratio is not None:
            max_drop = int(np.floor(total * self.max_ratio))
            max_drop = min(max_drop, max(total - self.min_keep, 0))
            if max_drop <= 0:
                flagged_pairs = []
            elif len(flagged_pairs) > max_drop:
                flagged_pairs = flagged_pairs[:max_drop]
        else:
            max_drop = total - self.min_keep
            if max_drop < 0:
                max_drop = 0
            if len(flagged_pairs) > max_drop:
                flagged_pairs = flagged_pairs[:max_drop]

        flagged_indices = [idx for (_, idx) in flagged_pairs]
        for idx in flagged_indices:
            mask[idx] = False

        stats = CleanerStats(
            total=total,
            removed=len(flagged_indices),
            threshold=threshold,
            median_distance=median_distance,
            mad=mad,
        )
        return mask, stats