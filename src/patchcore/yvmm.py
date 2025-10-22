"""Yarn voxel manifold mapping extensions for PatchCore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class YarnVoxelConfig:
    """Configuration container for the yarn voxel manifold module."""

    depth: int = 4
    hidden_channels: int = 64
    fusion_hidden: int = 128
    stability_eps: float = 1e-6
    encoded_scale: float = 1.0
    fold_scale: float = 1.0
    residual_mix: float = 1.0


class YarnVoxelizer(torch.nn.Module):
    """Lifts 2D patch embeddings into a lightweight 3D voxel volume."""

    def __init__(self, feature_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 to capture yarn trajectories")
        self.feature_dim = feature_dim
        self.depth = depth
        self.depth_projection = torch.nn.Linear(feature_dim, feature_dim * depth)

    def forward(
        self,
        flat_features: torch.Tensor,
        batchsize: int,
        patch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if batchsize <= 0:
            raise ValueError("batchsize must be positive for voxelization")

        patch_h, patch_w = patch_shape
        patch_count = patch_h * patch_w
        if patch_count <= 0:
            raise ValueError("patch_shape must describe a non-empty grid")

        expected = batchsize * patch_count
        if flat_features.shape[0] != expected:
            raise ValueError(
                "Flat features do not match patch grid dimensions: "
                f"expected {expected}, got {flat_features.shape[0]}"
            )

        projected = self.depth_projection(flat_features)
        projected = projected.view(batchsize, patch_count, self.depth, self.feature_dim)
        projected = projected.permute(0, 3, 2, 1).contiguous()
        voxels = projected.view(
            batchsize, self.feature_dim, self.depth, patch_h, patch_w
        )
        return voxels


class VoxelManifoldEncoder(torch.nn.Module):
    """Encodes voxel grids with a shallow 3D convolutional manifold encoder."""

    def __init__(self, feature_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(feature_dim, hidden_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv3d(hidden_channels, feature_dim, kernel_size=1),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(voxels)
        encoded = encoded.mean(dim=2)  # Average over depth.
        encoded = encoded.permute(0, 2, 3, 1).contiguous()
        batchsize, height, width, channels = encoded.shape
        return encoded.view(batchsize, height * width, channels)


class FoldMonitorProjector(torch.nn.Module):
    """Estimates geodesic fold energy over the voxelized yarn manifold."""

    def __init__(self, stability_eps: float = 1e-6) -> None:
        super().__init__()
        self.stability_eps = stability_eps

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        batchsize, channels, depth, height, width = voxels.shape
        voxels_patch = voxels.permute(0, 3, 4, 2, 1).contiguous()
        voxels_patch = voxels_patch.view(batchsize, height * width, depth, channels)

        centerline = voxels_patch.mean(dim=2, keepdim=True)
        variance = (voxels_patch - centerline).pow(2).mean(dim=2)
        boundary_span = (
            voxels_patch[:, :, -1, :] - voxels_patch[:, :, 0, :]
        ).abs()
        response = torch.sqrt(variance + boundary_span.pow(2) + self.stability_eps)
        return response


class YarnVoxelManifoldMapping(torch.nn.Module):
    """Augments patch embeddings with yarn voxel manifold reasoning."""

    def __init__(self, feature_dim: int, config: YarnVoxelConfig | None = None) -> None:
        super().__init__()
        self.config = config or YarnVoxelConfig()
        if self.config.residual_mix <= 0:
            raise ValueError("residual_mix must be positive to retain manifold signal")
        self.voxelizer = YarnVoxelizer(feature_dim, self.config.depth)
        self.manifold_encoder = VoxelManifoldEncoder(
            feature_dim, self.config.hidden_channels
        )
        self.fold_monitor = FoldMonitorProjector(self.config.stability_eps)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, self.config.fusion_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.fusion_hidden, feature_dim),
        )
        self.delta_activation = torch.nn.Tanh()
        self.register_buffer("last_residual_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_fold_energy", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        flat_features: torch.Tensor,
        patch_shape: Tuple[int, int],
        batchsize: int,
    ) -> torch.Tensor:
        if flat_features.ndim != 2:
            raise ValueError("Expected 2D tensor of flattened patch embeddings")

        voxels = self.voxelizer(flat_features, batchsize, patch_shape)
        encoded = self.manifold_encoder(voxels)
        fold_energy = self.fold_monitor(voxels)
        encoded = encoded * self.config.encoded_scale
        fold_energy = fold_energy * self.config.fold_scale
        combined = torch.cat([encoded, fold_energy], dim=-1)
        delta = self.fusion(combined.view(-1, combined.shape[-1]))
        delta = self.delta_activation(delta)
        delta = delta.view_as(flat_features) * self.config.residual_mix
        with torch.no_grad():
            residual_norm = delta.norm(dim=1).mean()
            fold_mean = fold_energy.norm(dim=-1).mean()
            self.last_residual_norm.copy_(residual_norm.detach())
            self.last_fold_energy.copy_(fold_mean.detach())
        return flat_features + delta