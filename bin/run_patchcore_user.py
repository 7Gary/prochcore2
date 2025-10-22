import contextlib
import logging
import os
import sys
import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils


LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
             "custom": ["patchcore.datasets.custom", "CustomDataset"]
             }



@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])
                outlier_stats = getattr(PatchCore, "training_outlier_stats", {})
                contamination_ratio = outlier_stats.get("pseudo_ratio", 0.0) * 100
                recovered_patches = outlier_stats.get("recovered_patches", 0)
                if contamination_ratio > 0:
                    LOGGER.info(
                        "训练提示: 约 %.2f%% 的训练图像被判定为伪异常, 这是漏检率和过杀率上升的主要原因。",
                        contamination_ratio,
                    )
                if recovered_patches > 0:
                    LOGGER.info(
                        "通过伪异常清洗机制已回收 %d 个近似正常补丁用于记忆库。",
                        recovered_patches,
                    )
                pseudo_samples = outlier_stats.get("pseudo_metadata", [])
                if pseudo_samples:
                    sample_paths = [
                        meta.get("image_path") or meta.get("image_name")
                        for meta in pseudo_samples[:5]
                    ]
                    LOGGER.info(
                        "伪异常样本示例: %s",
                        ", ".join(filter(None, sample_paths)),
                    )

            torch.cuda.empty_cache()
            aggregator = {
                "scores": [],
                "segmentations": [],
                "image_names": None,
                "image_paths": None,
            }
            labels_gt = None
            masks_gt = None
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                (
                    scores,
                    segmentations,
                    labels_pred,
                    masks_pred,
                    image_names,
                    image_paths,
                ) = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                if aggregator["image_names"] is None:
                    aggregator["image_names"] = image_names
                    aggregator["image_paths"] = image_paths
                if labels_gt is None:
                    labels_gt = labels_pred
                    masks_gt = masks_pred

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            image_names = np.array(aggregator.get("image_names") or [], dtype=object)
            image_paths = np.array(aggregator.get("image_paths") or [], dtype=object)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )
            reshaped_scores = scores.reshape(-1, 2)

            # 使用最大值或平均值聚合
            scores = np.max(reshaped_scores, axis=1)  # 或 np.mean

            if image_paths.size:
                image_paths = image_paths.reshape(-1, 2)
                base_image_paths = [pair[0] for pair in image_paths.tolist()]
            else:
                base_image_paths = [None] * len(scores)
            if image_names.size:
                image_names = image_names.reshape(-1, 2)
                base_image_names = [pair[0] for pair in image_names.tolist()]
            else:
                base_image_names = [None] * len(scores)

            LOGGER.info("Computing evaluation metrics.")
            imagewise_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )

            auroc = imagewise_metrics["auroc"]

            # 计算漏检率和过杀率
            fpr = imagewise_metrics["fpr"]
            tpr = imagewise_metrics["tpr"]
            thresholds = imagewise_metrics["threshold"]

            # 找到最佳阈值（Youden's J statistic）
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]

            # 使用最佳阈值进行预测
            y_pred = (scores >= best_threshold).astype(int)
            y_true = np.array(anomaly_labels)

            # 计算混淆矩阵
            tn = np.sum((y_pred == 0) & (y_true == 0))  # 真阴性
            fp = np.sum((y_pred == 1) & (y_true == 0))  # 假阳性
            fn = np.sum((y_pred == 0) & (y_true == 1))  # 假阴性
            tp = np.sum((y_pred == 1) & (y_true == 1))  # 真阳性

            # 计算漏检率和过杀率
            try:
                miss_rate = fn / (tp + fn)  # 漏检率 = FN / (TP+FN)
            except ZeroDivisionError:
                miss_rate = float('nan')

            try:
                overkill_rate = fp / (tn + fp)  # 过杀率 = FP / (TN+FP)
            except ZeroDivisionError:
                overkill_rate = float('nan')

            LOGGER.info("Best Threshold: {:.4f}".format(best_threshold))
            LOGGER.info("Miss Rate (漏检率): {:.2%}".format(miss_rate))
            LOGGER.info("Overkill Rate (过杀率): {:.2%}".format(overkill_rate))

            predicted_anomaly_indices = np.where(y_pred == 1)[0]
            predicted_records = []
            for idx in predicted_anomaly_indices:
                path = base_image_paths[idx]
                name = base_image_names[idx]
                if path:
                    predicted_records.append(path)
                elif name:
                    predicted_records.append(name)
                else:
                    predicted_records.append(f"index_{idx}")

            # if predicted_records:
            #     LOGGER.info(
            #         "预测为异常的样本路径列表: %s",
            #         " | ".join(predicted_records),
            #     )
            # else:
            #     LOGGER.info("预测为异常的样本路径列表: 无")

            contamination_ratio = 0.0
            recovered_patches = 0
            if PatchCore_list:
                outlier_stats = getattr(PatchCore_list[0], "training_outlier_stats", {})
                contamination_ratio = outlier_stats.get("pseudo_ratio", 0.0) * 100
                recovered_patches = outlier_stats.get("recovered_patches", 0)
            # LOGGER.info(
            #     "诊断: 当前训练集污染率约为 %.2f%%, 伪异常清洗后回收补丁数为 %d, 这是影响漏检率/过杀率的关键因素。",
            #     contamination_ratio,
            #     recovered_patches,
            # )

            # Compute PRO score & PW Auroc for all images
            # pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            #     segmentations, masks_gt
            # )
            # 修改后（跳过像素级评估）
            # full_pixel_auroc = pixel_scores["auroc"]

            full_pixel_auroc = -1.0  # 占位值

            # Compute PRO score & PW Auroc only images with anomalies
            # sel_idxs = []
            # for i in range(len(masks_gt)):
            #     if np.sum(masks_gt[i]) > 0:
            #         sel_idxs.append(i)
            # pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            #     [segmentations[i] for i in sel_idxs],
            #     [masks_gt[i] for i in sel_idxs],
            # )
            # anomaly_pixel_auroc = pixel_scores["auroc"]

            anomaly_pixel_auroc = -1.0

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
@click.option("--use_pseudo_negatives", is_flag=True, help="Enable pseudo-negative memory bank.")
@click.option(
    "--pseudo_negative_ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of training images to treat as pseudo anomalies.",
)
@click.option(
    "--pseudo_negative_min_count",
    type=int,
    default=1,
    show_default=True,
    help="Minimum number of pseudo anomalies to keep.",
)
@click.option(
    "--pseudo_negative_max_count",
    type=int,
    default=None,
    help="Optional cap on pseudo anomaly images.",
)
@click.option(
    "--pseudo_negative_temperature",
    type=float,
    default=0.05,
    show_default=True,
    help="Temperature for converting pseudo-memory distances to scores.",
)
@click.option(
    "--pseudo_negative_weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight of the pseudo memory branch when combining scores.",
)
@click.option(
    "--pseudo_negative_eps",
    type=float,
    default=1e-6,
    show_default=True,
    help="Stability term for Mahalanobis covariance inversion.",
)
@click.option(
    "--normal_memory_weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight of the normal memory branch.",
)
@click.option(
    "--clean_patch_quantile",
    type=float,
    default=0.9,
    show_default=True,
    help="Quantile used to keep clean-looking patches from contaminated images.",
)
@click.option(
    "--clean_patch_min_ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum ratio of clean patches required to recycle a contaminated image.",
)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    use_pseudo_negatives,
    pseudo_negative_ratio,
    pseudo_negative_min_count,
    pseudo_negative_max_count,
    pseudo_negative_temperature,
    pseudo_negative_weight,
    pseudo_negative_eps,
    normal_memory_weight,
    clean_patch_quantile,
    clean_patch_min_ratio,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                use_pseudo_negatives=use_pseudo_negatives,
                pseudo_negative_ratio=pseudo_negative_ratio,
                pseudo_negative_min_count=pseudo_negative_min_count,
                pseudo_negative_max_count=pseudo_negative_max_count,
                pseudo_negative_temperature=pseudo_negative_temperature,
                pseudo_negative_weight=pseudo_negative_weight,
                pseudo_negative_eps=pseudo_negative_eps,
                normal_memory_weight=normal_memory_weight,
                clean_patch_quantile=clean_patch_quantile,
                clean_patch_min_ratio=clean_patch_min_ratio,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()