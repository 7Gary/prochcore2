"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

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

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

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

        (
            normal_features,
            pseudo_anomaly_features,
            outlier_stats,
        ) = self._separate_normal_and_pseudo_anomalies(
            features_per_image, image_level_embeddings, metadata_per_image
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
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

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
        normal_patch_scores, _, _ = self.anomaly_scorer.predict([features])
        combined_patch_scores = normal_patch_scores

        if self._pseudo_memory_active and self.pseudo_anomaly_scorer is not None:
            _, pseudo_distances, _ = self.pseudo_anomaly_scorer.predict([features])
            pseudo_component = np.exp(
                -np.maximum(pseudo_distances.mean(axis=-1), 0)
                / max(self.pseudo_negative_temperature, 1e-12)
            )
            combined_patch_scores = (
                self.normal_memory_weight * normal_patch_scores
                + self.pseudo_negative_weight * pseudo_component
            )

        return combined_patch_scores, combined_patch_scores

    def _separate_normal_and_pseudo_anomalies(
        self, features_per_image, image_level_embeddings, metadata_per_image
    ):
        """Split training features into normal/pseudo-anomalous banks.

        Following the pseudo-outlier filtering strategies proposed in
        CVPR'22 POF and the dual-memory modelling ideas exemplified by
        AAAI'22 MemSeg, we treat high Mahalanobis-distance samples as
        pseudo negatives and store them separately from the normal memory
        bank.
        """
        if not features_per_image:
            raise RuntimeError("No features were computed for memory bank training.")

        normal_features = np.concatenate(features_per_image, axis=0)
        pseudo_anomaly_features = None
        outlier_stats = {
            "total_images": len(features_per_image),
            "pseudo_count": 0,
            "pseudo_ratio": 0.0,
            "threshold": None,
            "pseudo_metadata": [],
        }

        if not self.use_pseudo_negatives:
            return normal_features, pseudo_anomaly_features, outlier_stats

        image_embeddings = np.asarray(image_level_embeddings)
        mean_embedding = np.mean(image_embeddings, axis=0)
        centered_embeddings = image_embeddings - mean_embedding
        covariance = np.cov(centered_embeddings, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * self.pseudo_negative_eps
        inv_covariance = np.linalg.pinv(covariance)
        mahalanobis = np.sqrt(
            np.sum(centered_embeddings @ inv_covariance * centered_embeddings, axis=1)
        )

        num_candidates = len(mahalanobis)
        desired_pseudo = int(num_candidates * self.pseudo_negative_ratio)
        desired_pseudo = max(desired_pseudo, self.pseudo_negative_min_count)
        if self.pseudo_negative_max_count is not None:
            desired_pseudo = min(desired_pseudo, self.pseudo_negative_max_count)
        desired_pseudo = min(desired_pseudo, num_candidates - 1)

        if desired_pseudo <= 0:
            return normal_features, pseudo_anomaly_features, outlier_stats

        sorted_indices = np.argsort(mahalanobis)[::-1]
        pseudo_indices = sorted_indices[:desired_pseudo]
        mask = np.ones(num_candidates, dtype=bool)
        mask[pseudo_indices] = False

        normal_features = np.concatenate(
            [features_per_image[idx] for idx in np.where(mask)[0]], axis=0
        )
        pseudo_anomaly_features = np.concatenate(
            [features_per_image[idx] for idx in pseudo_indices], axis=0
        )

        outlier_stats.update(
            {
                "pseudo_count": len(pseudo_indices),
                "pseudo_ratio": len(pseudo_indices) / float(num_candidates),
                "threshold": float(mahalanobis[pseudo_indices[-1]]),
                "pseudo_metadata": [metadata_per_image[idx] for idx in pseudo_indices],
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

        return normal_features, pseudo_anomaly_features, outlier_stats


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