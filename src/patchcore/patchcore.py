"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore.contamination as contamination

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.yvmm
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.contamination_controller = None
        self.closed_loop_controller = None
        self.dual_memory = contamination.ReciprocalDualMemoryDistillation()
        self.dashboard = contamination.ContaminationDashboard()
        self._current_gamma = 0.0
        self.yvmm_module = None

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.normal_memory_weight = kwargs.pop("normal_memory_weight", 1.0)

        self.use_pseudo_negatives = kwargs.pop("use_pseudo_negatives", False)
        self.pseudo_negative_ratio = kwargs.pop("pseudo_negative_ratio", 0.05)
        self.pseudo_negative_min_count = kwargs.pop("pseudo_negative_min_count", 1)
        self.pseudo_negative_max_count = kwargs.pop(
            "pseudo_negative_max_count", None
        )
        self.pseudo_negative_temperature = kwargs.pop(
            "pseudo_negative_temperature", 0.05
        )
        self.pseudo_negative_weight = kwargs.pop("pseudo_negative_weight", 1.0)
        self.pseudo_negative_eps = kwargs.pop("pseudo_negative_eps", 1e-6)
        self.clean_patch_quantile = kwargs.pop("clean_patch_quantile", 0.9)
        self.clean_patch_min_ratio = kwargs.pop("clean_patch_min_ratio", 0.05)

        self.enable_contamination_elasticity = kwargs.pop(
            "enable_contamination_elasticity", True
        )
        self.elasticity_smoothing = kwargs.pop("elasticity_smoothing", 0.2)
        self.elasticity_temperature = kwargs.pop("elasticity_temperature", 2.0)
        self.elasticity_min_threshold = kwargs.pop("elasticity_min_threshold", 1.0)
        self.dual_memory_temperature = kwargs.pop(
            "dual_memory_temperature", self.pseudo_negative_temperature
        )
        self.enable_closed_loop = kwargs.pop("enable_closed_loop", True)
        self.closed_loop_alpha = kwargs.pop("closed_loop_alpha", 1.2)
        self.closed_loop_beta = kwargs.pop("closed_loop_beta", 0.5)
        self.closed_loop_eta = kwargs.pop("closed_loop_eta", 0.1)
        self.closed_loop_base_temperature = kwargs.pop(
            "closed_loop_base_temperature", self.pseudo_negative_temperature
        )
        self.closed_loop_max_temperature = kwargs.pop("closed_loop_max_temperature", 0.5)
        if self.closed_loop_max_temperature < self.closed_loop_base_temperature:
            self.closed_loop_max_temperature = self.closed_loop_base_temperature

        self.pseudo_anomaly_scorer = None
        self._pseudo_memory_active = False
        if self.use_pseudo_negatives:
            pseudo_nn_method = type(nn_method)(
                getattr(nn_method, "on_gpu", False), getattr(nn_method, "num_workers", 4)
            )
            self.pseudo_anomaly_scorer = patchcore.common.NearestNeighbourScorer(
                n_nearest_neighbours=anomaly_score_num_nn,
                nn_method=pseudo_nn_method,
            )

        self.training_outlier_stats = {}

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.dual_memory = contamination.ReciprocalDualMemoryDistillation(
            temperature=self.dual_memory_temperature
        )
        self.dashboard = contamination.ContaminationDashboard()
        self._current_gamma = 0.0
        self.contamination_controller = None
        self.closed_loop_controller = None
        yvmm_config = kwargs.pop("yvmm_config", None)
        if yvmm_config is not None:
            if isinstance(yvmm_config, dict):
                config = patchcore.yvmm.YarnVoxelConfig(**yvmm_config)
            elif isinstance(yvmm_config, patchcore.yvmm.YarnVoxelConfig):
                config = yvmm_config
            else:
                raise TypeError(
                    "yvmm_config must be a dict or YarnVoxelConfig instance"
                )
            self.yvmm_module = patchcore.yvmm.YarnVoxelManifoldMapping(
                self.target_embed_dimension, config
            ).to(self.device)
            self.forward_modules["yvmm_module"] = self.yvmm_module
        if self.enable_contamination_elasticity:
            self.contamination_controller = contamination.ContaminationElasticityController(
                feature_dim=self.target_embed_dimension,
                smoothing=self.elasticity_smoothing,
                elasticity_temperature=self.elasticity_temperature,
                minimum_threshold=self.elasticity_min_threshold,
            )
            if self.enable_closed_loop:
                self.closed_loop_controller = contamination.ClosedLoopContaminationControl(
                    self.contamination_controller,
                    alpha=self.closed_loop_alpha,
                    beta=self.closed_loop_beta,
                    eta=self.closed_loop_eta,
                    base_temperature=self.closed_loop_base_temperature,
                    max_temperature=self.closed_loop_max_temperature,
                )

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if self.yvmm_module is not None:
            batchsize = images.shape[0]
            features = self.yvmm_module(features, patch_shapes[0], batchsize)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)
        if self.yvmm_module is not None and hasattr(self.yvmm_module, "last_residual_norm"):
            try:
                residual = float(self.yvmm_module.last_residual_norm.item())
                fold_energy = float(self.yvmm_module.last_fold_energy.item())
                LOGGER.info(
                    "YVMM 诊断: 残差范数 %.4f, 折叠能量 %.4f. 可通过调整"
                    " --yvmm_residual_mix / --yvmm_fold_scale 优化。",
                    residual,
                    fold_energy,
                )
            except (RuntimeError, AttributeError):
                LOGGER.debug("Failed to read YVMM diagnostics after training.")

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features_per_image = []
        image_level_embeddings = []
        metadata_per_image = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                batch_images = batch
                batch_metadata = {}
                if isinstance(batch, dict):
                    batch_images = batch["image"]
                    batch_metadata = {
                        key: batch.get(key)
                        for key in ["image_path", "image_name", "anomaly"]
                    }

                batch_size = batch_images.shape[0]
                for idx in range(batch_size):
                    single_image = batch_images[idx: idx + 1]
                    single_features = np.asarray(_image_to_features(single_image))
                    features_per_image.append(single_features)
                    image_level_embeddings.append(
                        np.mean(single_features, axis=0)
                    )

                    metadata = {
                        "image_path": None,
                        "image_name": None,
                        "anomaly": None,
                    }
                    for key, values in batch_metadata.items():
                        if values is None:
                            continue
                        if isinstance(values, (list, tuple)):
                            metadata[key] = values[idx]
                        else:
                            metadata[key] = values
                    metadata_per_image.append(metadata)
        elasticity_state = None
        if (
            self.contamination_controller is not None
            and len(image_level_embeddings) > 0
        ):
            stacked_embeddings = np.asarray(image_level_embeddings)
            elasticity_state = self.contamination_controller.estimate(
                stacked_embeddings
            )
            self._current_gamma = elasticity_state.gamma
            self.dashboard.log_elasticity(elasticity_state)
        else:
            self._current_gamma = 0.0

        (
            normal_features,
            pseudo_anomaly_features,
            outlier_stats,
        ) = self._separate_normal_and_pseudo_anomalies(
            features_per_image,
            image_level_embeddings,
            metadata_per_image,
            elasticity_state=elasticity_state,
        )

        features = self.featuresampler.run(normal_features)

        self.anomaly_scorer.fit(detection_features=[features])

        self._pseudo_memory_active = False
        if (
            self.use_pseudo_negatives
            and pseudo_anomaly_features is not None
            and len(pseudo_anomaly_features) > 0
            and self.pseudo_anomaly_scorer is not None
        ):
            self.pseudo_anomaly_scorer.fit(
                detection_features=[pseudo_anomaly_features]
            )
            self._pseudo_memory_active = True

        self.training_outlier_stats = outlier_stats
        if elasticity_state is not None:
            if self.closed_loop_controller is not None:
                loop_state = self.closed_loop_controller.step(
                    elasticity_state.contamination_rate
                )
                self.normal_memory_weight = loop_state.normal_weight
                self.pseudo_negative_weight = loop_state.pseudo_weight
                self.pseudo_negative_temperature = loop_state.temperature
                self.dual_memory.temperature = loop_state.temperature
                self.dashboard.log_closed_loop(loop_state)
                outlier_stats.update(
                    {
                        "closed_loop_gain": float(loop_state.control_gain),
                        "closed_loop_integral": float(loop_state.integral_term),
                        "controlled_normal_weight": float(
                            self.normal_memory_weight
                        ),
                        "controlled_pseudo_weight": float(
                            self.pseudo_negative_weight
                        ),
                        "controlled_temperature": float(self.dual_memory.temperature),
                    }
                )
            outlier_stats.update(
                {
                    "elasticity": float(elasticity_state.elasticity),
                    "gamma": float(elasticity_state.gamma),
                    "contamination_rate": float(
                        elasticity_state.contamination_rate
                    ),
                }
            )

        outlier_stats["dashboard"] = self.dashboard.export()
        if outlier_stats.get("pseudo_ratio", 0) > 0:
            LOGGER.info(
                "Training set contamination estimate: %.2f%% pseudo anomalies detected.",
                outlier_stats["pseudo_ratio"] * 100.0,
            )

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        image_names = []
        image_paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                batch_names = None
                batch_paths = None
                if isinstance(batch, dict):
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())
                    batch_names = batch.get("image_name")
                    batch_paths = batch.get("image_path")
                    images = batch["image"]
                else:
                    images = batch

                _scores, _masks = self._predict(images)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
                batchsize = len(_scores)

                def _extend_metadata(container, values):
                    if values is None:
                        container.extend([None] * batchsize)
                    elif isinstance(values, (list, tuple)):
                        container.extend(list(values))
                    else:
                        container.append(values)

                _extend_metadata(image_names, batch_names)
                _extend_metadata(image_paths, batch_paths)

        return scores, masks, labels_gt, masks_gt, image_names, image_paths

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores, image_scores = self._compute_dual_memory_scores(features)
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def _compute_dual_memory_scores(self, features):
        (
            normal_patch_scores,
            normal_distances,
            _,
        ) = self.anomaly_scorer.predict([features])

        pseudo_distances = None

        if self._pseudo_memory_active and self.pseudo_anomaly_scorer is not None:
            _, pseudo_distances, _ = self.pseudo_anomaly_scorer.predict([features])
        combined_patch_scores = self.dual_memory.combine(
            normal_patch_scores,
            normal_distances,
            pseudo_distances,
            gamma=self._current_gamma,
            normal_weight=self.normal_memory_weight,
            pseudo_weight=self.pseudo_negative_weight,
        )
        self.dashboard.log_dual_memory(self.dual_memory.energy_trace)

        return combined_patch_scores, combined_patch_scores

    def _separate_normal_and_pseudo_anomalies(
            self,
            features_per_image,
            image_level_embeddings,
            metadata_per_image,
            elasticity_state=None,
    ):
        """Split training features into normal/pseudo-anomalous banks.

        This routine integrates the proposed contamination elasticity principle
        by default. When ``elasticity_state`` is provided, the robust Mahalanobis
        distances and elastic threshold are used to identify pseudo anomalies;
        otherwise, the classical Mahalanobis heuristic is used as a fallback.
        """
        if not features_per_image:
            raise RuntimeError("No features were computed for memory bank training.")
        feature_dim = features_per_image[0].shape[-1]
        normal_features = np.concatenate(features_per_image, axis=0)
        pseudo_anomaly_features = None
        outlier_stats = {
            "total_images": len(features_per_image),
            "pseudo_count": 0,
            "pseudo_ratio": 0.0,
            "threshold": None,
            "pseudo_metadata": [],
            "initial_pseudo_count": 0,
            "recovered_images": 0,
            "recovered_patches": 0,
            "pseudo_patch_threshold": None,
        }

        if not self.use_pseudo_negatives:
            return normal_features, pseudo_anomaly_features, outlier_stats

        image_embeddings = np.asarray(image_level_embeddings)
        if elasticity_state is not None:
            robust_distances = np.asarray(elasticity_state.distances)
        else:
            mean_embedding = np.mean(image_embeddings, axis=0)
            centered_embeddings = image_embeddings - mean_embedding
            covariance = np.cov(centered_embeddings, rowvar=False)
            covariance += np.eye(covariance.shape[0]) * self.pseudo_negative_eps
            inv_covariance = np.linalg.pinv(covariance)
            robust_distances = np.sqrt(
                np.sum(centered_embeddings @ inv_covariance * centered_embeddings, axis=1)
            )

        num_candidates = len(robust_distances)
        if num_candidates == 0:
            return normal_features, pseudo_anomaly_features, outlier_stats


        desired_pseudo = int(num_candidates * self.pseudo_negative_ratio)
        desired_pseudo = max(desired_pseudo, self.pseudo_negative_min_count)
        if self.pseudo_negative_max_count is not None:
            desired_pseudo = min(desired_pseudo, self.pseudo_negative_max_count)
        desired_pseudo = min(desired_pseudo, num_candidates - 1)

        if desired_pseudo <= 0:
            return normal_features, pseudo_anomaly_features, outlier_stats

        sorted_indices = np.argsort(robust_distances)[::-1]
        if elasticity_state is not None:
            pseudo_mask = np.asarray(elasticity_state.pseudo_mask)
            pseudo_indices = np.where(pseudo_mask)[0]
            if desired_pseudo > 0 and len(pseudo_indices) < desired_pseudo:
                existing = set(pseudo_indices.tolist())
                padding = [
                    idx for idx in sorted_indices if idx not in existing
                ][: desired_pseudo - len(pseudo_indices)]
                if padding:
                    pseudo_indices = np.concatenate([pseudo_indices, padding])
        else:
            pseudo_indices = sorted_indices[:desired_pseudo]
        outlier_stats["initial_pseudo_count"] = len(pseudo_indices)

        if len(pseudo_indices) == 0:
            return normal_features, pseudo_anomaly_features, outlier_stats
        mask = np.ones(num_candidates, dtype=bool)
        mask[pseudo_indices] = False

        normal_indices = np.where(mask)[0]
        normal_feature_list = [features_per_image[idx] for idx in normal_indices]
        pseudo_feature_list = [features_per_image[idx] for idx in pseudo_indices]
        pseudo_metadata = [metadata_per_image[idx] for idx in pseudo_indices]

        if not normal_feature_list:
            LOGGER.warning(
                "All training samples were flagged as pseudo anomalies; disabling pseudo memory bank."
            )
            return normal_features, None, outlier_stats

        normal_features = np.concatenate(normal_feature_list, axis=0)
        pseudo_anomaly_features = (
            np.concatenate(pseudo_feature_list, axis=0)
            if pseudo_feature_list
            else None
        )
        (
            normal_features,
            pseudo_anomaly_features,
            recovery_stats,
            filtered_metadata,
        ) = self._recover_clean_patches_from_contaminated_images(
            normal_feature_list,
            pseudo_feature_list,
            pseudo_metadata,
            feature_dim,
        )
        outlier_stats.update(recovery_stats)

        outlier_stats.update(
            {
                "pseudo_metadata": filtered_metadata,
                "pseudo_count": len(filtered_metadata),
                "pseudo_ratio": len(filtered_metadata) / float(num_candidates)
                if num_candidates > 0
                else 0.0,
                "threshold": float(
                    elasticity_state.threshold
                    if elasticity_state is not None
                    else robust_distances[pseudo_indices[-1]]
                ),
                "distance_mean": float(np.mean(robust_distances)),
                "distance_std": float(np.std(robust_distances)),
                "robust_distance_topk": robust_distances[
                    sorted_indices[: min(20, len(sorted_indices))]
                ].tolist(),
            }
        )

        LOGGER.info(
            "Identified %d pseudo-anomalous images out of %d (%.2f%%) during training.",
            outlier_stats["pseudo_count"],
            outlier_stats["total_images"],
            outlier_stats["pseudo_ratio"] * 100.0,
        )

        if outlier_stats["pseudo_metadata"]:
            sample_meta = outlier_stats["pseudo_metadata"][:5]
            sample_names = [
                meta.get("image_name") or meta.get("image_path") for meta in sample_meta
            ]
            LOGGER.info(
                "Example pseudo-anomaly candidates: %s",
                ", ".join(filter(None, sample_names)) or "N/A",
            )
        if outlier_stats.get("recovered_patches", 0) > 0:
            LOGGER.info(
                "Recovered %d clean-looking patches from %d pseudo-anomaly images (quantile %.2f).",
                outlier_stats["recovered_patches"],
                outlier_stats.get("recovered_images", 0),
                outlier_stats.get("pseudo_patch_threshold", 0.0),
            )

        return normal_features, pseudo_anomaly_features, outlier_stats

    def _recover_clean_patches_from_contaminated_images(
        self,
        normal_feature_list,
        pseudo_feature_list,
        pseudo_metadata,
        feature_dim,
    ):
        if not pseudo_feature_list:
            clean_bank = (
                np.concatenate(normal_feature_list, axis=0)
                if normal_feature_list
                else np.zeros((0, feature_dim))
            )
            return clean_bank, None, {
                "recovered_images": 0,
                "recovered_patches": 0,
                "pseudo_patch_threshold": None,
            }, []

        normal_bank = (
            np.concatenate(normal_feature_list, axis=0)
            if normal_feature_list
            else np.zeros((0, feature_dim))
        )

        if normal_bank.shape[0] < 2:
            pseudo_bank = (
                np.concatenate(pseudo_feature_list, axis=0)
                if pseudo_feature_list
                else None
            )
            return normal_bank, pseudo_bank, {
                "recovered_images": 0,
                "recovered_patches": 0,
                "pseudo_patch_threshold": None,
            }, pseudo_metadata

        mean_normal = np.mean(normal_bank, axis=0)
        centered = normal_bank - mean_normal
        covariance = np.cov(centered, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * self.pseudo_negative_eps
        inv_covariance = np.linalg.pinv(covariance)

        base_distances = self._mahalanobis_distance(normal_bank, mean_normal, inv_covariance)
        threshold = float(
            np.quantile(base_distances, np.clip(self.clean_patch_quantile, 0.5, 0.99))
        )

        clean_collections = [normal_bank]
        pseudo_collections = []
        retained_metadata = []
        recovered_images = 0
        recovered_patches = 0

        for features, metadata in zip(pseudo_feature_list, pseudo_metadata):
            if features.size == 0:
                continue
            distances = self._mahalanobis_distance(features, mean_normal, inv_covariance)
            clean_mask = distances <= threshold
            clean_count = int(np.sum(clean_mask))
            total_count = features.shape[0]

            if total_count == 0:
                continue

            clean_ratio = clean_count / float(total_count)
            if clean_count > 0 and clean_ratio >= self.clean_patch_min_ratio:
                clean_collections.append(features[clean_mask])
                recovered_images += 1
                recovered_patches += clean_count

            anomaly_mask = ~clean_mask if clean_count else np.ones_like(clean_mask, dtype=bool)
            anomaly_count = int(np.sum(anomaly_mask))
            if anomaly_count > 0:
                pseudo_collections.append(features[anomaly_mask])
                retained_metadata.append(metadata)

        clean_bank = np.concatenate(clean_collections, axis=0)
        pseudo_bank = (
            np.concatenate(pseudo_collections, axis=0)
            if pseudo_collections
            else None
        )

        stats = {
            "recovered_images": recovered_images,
            "recovered_patches": recovered_patches,
            "pseudo_patch_threshold": threshold,
        }

        return clean_bank, pseudo_bank, stats, retained_metadata

    @staticmethod
    def _mahalanobis_distance(features, mean_vector, inv_covariance):
        centered = features - mean_vector
        return np.sqrt(np.sum(centered @ inv_covariance * centered, axis=1))


    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "use_pseudo_negatives": self.use_pseudo_negatives,
            "pseudo_negative_ratio": self.pseudo_negative_ratio,
            "pseudo_negative_min_count": self.pseudo_negative_min_count,
            "pseudo_negative_max_count": self.pseudo_negative_max_count,
            "pseudo_negative_temperature": self.pseudo_negative_temperature,
            "pseudo_negative_weight": self.pseudo_negative_weight,
            "pseudo_negative_eps": self.pseudo_negative_eps,
            "normal_memory_weight": self.normal_memory_weight,
            "clean_patch_quantile": self.clean_patch_quantile,
            "clean_patch_min_ratio": self.clean_patch_min_ratio,
            "enable_contamination_elasticity": self.enable_contamination_elasticity,
            "elasticity_smoothing": self.elasticity_smoothing,
            "elasticity_temperature": self.elasticity_temperature,
            "elasticity_min_threshold": self.elasticity_min_threshold,
            "dual_memory_temperature": self.dual_memory.temperature,
            "enable_closed_loop": self.enable_closed_loop,
            "closed_loop_alpha": self.closed_loop_alpha,
            "closed_loop_beta": self.closed_loop_beta,
            "closed_loop_eta": self.closed_loop_eta,
            "closed_loop_base_temperature": self.closed_loop_base_temperature,
            "closed_loop_max_temperature": self.closed_loop_max_temperature,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
