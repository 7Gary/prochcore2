import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
    heatmap_cmap="inferno",
    heatmap_alpha=0.6,
    show_colorbar=True,
    file_suffix="_viz",
    file_extension=None,
    dpi=200,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
        heatmap_cmap: [str] Matplotlib colormap name used for heatmap rendering.
        heatmap_alpha: [float] Alpha value when blending heatmap with the image.
        show_colorbar: [bool] If true, attaches a colorbar for the heatmap scale.
        file_suffix: [str] Optional suffix appended to each savename.
        file_extension: [str] Force a specific extension (e.g., ".png").
        dpi: [int] Dots per inch resolution used when saving the figure.

    Returns:
        List[str]: Absolute file paths of the saved visualization figures.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    def _normalize_map(segmentation_map):
        segmentation_map = np.asarray(segmentation_map).astype(np.float32)
        segmentation_map = np.nan_to_num(segmentation_map, nan=0.0, posinf=0.0, neginf=0.0)
        segmentation_map -= segmentation_map.min()
        max_value = segmentation_map.max()
        if max_value > 0:
            segmentation_map /= max_value
        return np.clip(segmentation_map, 0.0, 1.0)

    saved_paths = []

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
        normalized_segmentation = _normalize_map(segmentation)
        seg_height, seg_width = normalized_segmentation.shape[:2]
        colormap = plt.get_cmap(heatmap_cmap)
        heatmap_rgba = colormap(normalized_segmentation)
        heatmap_rgb = heatmap_rgba[..., :3]

        image_rgb = image.transpose(1, 2, 0).astype(np.float32)
        if image_rgb.shape[2] == 1:
            image_rgb = np.repeat(image_rgb, 3, axis=2)
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

        img_height, img_width = image_rgb.shape[:2]
        if (seg_height, seg_width) != (img_height, img_width):
            seg_image = PIL.Image.fromarray(
                (normalized_segmentation * 255.0).astype(np.uint8)
            ).resize((img_width, img_height), resample=PIL.Image.BILINEAR)
            normalized_segmentation = np.asarray(seg_image, dtype=np.float32) / 255.0
            heatmap_rgba = colormap(normalized_segmentation)
            heatmap_rgb = heatmap_rgba[..., :3]

        overlay = (1.0 - heatmap_alpha) * image_rgb + heatmap_alpha * heatmap_rgb
        overlay = np.clip(overlay, 0.0, 1.0)

        savename = image_path.split("/")
        savename = "_".join(filter(None, savename[-save_depth:]))
        savename = os.path.join(savefolder, savename)
        base_name, current_ext = os.path.splitext(savename)
        if file_suffix:
            base_name = base_name + file_suffix
        if file_extension is not None:
            enforced_ext = file_extension
            if not enforced_ext.startswith("."):
                enforced_ext = "." + enforced_ext
            savename = base_name + enforced_ext
        else:
            if not current_ext:
                savename = base_name + ".png"
            else:
                savename = base_name + current_ext
        num_cols = 3 + int(masks_provided)
        f, axes = plt.subplots(1, num_cols)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        else:
            axes = np.atleast_1d(axes)

        column_index = 0
        axes[column_index].imshow(image_rgb)
        axes[column_index].set_title("Input Image")
        column_index += 1

        if masks_provided:
            mask_display = mask
            if mask_display.ndim == 3:
                if mask_display.shape[0] in {1, 3} and mask_display.shape[0] != mask_display.shape[2]:
                    mask_display = mask_display.transpose(1, 2, 0)
                if mask_display.shape[2] == 1:
                    mask_display = np.repeat(mask_display, 3, axis=2)
            if mask_display.ndim == 2:
                axes[column_index].imshow(mask_display, cmap="gray")
            else:
                axes[column_index].imshow(mask_display)
            axes[column_index].set_title("Ground Truth")
            column_index += 1

        heatmap_axis = axes[column_index]
        im = heatmap_axis.imshow(normalized_segmentation, cmap=heatmap_cmap)
        heatmap_axis.set_title("Anomaly Heatmap")
        column_index += 1

        overlay_axis = axes[column_index]
        overlay_axis.imshow(overlay)
        overlay_axis.set_title("Overlay")

        if show_colorbar:
            f.colorbar(im, ax=heatmap_axis, fraction=0.046, pad=0.04)

        if anomaly_score in (None, "-1"):
            anomaly_score_text = "N/A"
        else:
            try:
                anomaly_score_text = f"{float(anomaly_score):.4f}"
            except (TypeError, ValueError):
                anomaly_score_text = str(anomaly_score)
        f.suptitle(f"Anomaly Score: {anomaly_score_text}", fontsize=12)

        for axis in axes:
            axis.axis("off")

        f.set_size_inches(3.5 * num_cols, 3.5)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig(savename, dpi=dpi)
        plt.close(f)

        saved_paths.append(savename)

    return saved_paths


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
