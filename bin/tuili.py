import contextlib
import gc
import logging
import os
import sys
import matplotlib.pyplot as plt

import click
import numpy as np
import torch
from PIL import Image

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


# 新增：单图推理命令
@main.command("single_inference")
@click.option("--image_path", type=str, required=True, help="Path to input image")
@click.option("--show_result", is_flag=True, help="Show segmentation result")
def single_inference(image_path, show_result):
    """新增单图推理功能"""
    # 获取预处理参数（假设使用MVTec的预处理）
    dataset_info = _DATASETS["mvtec"]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    # 创建单图数据集
    class SingleImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_path, resize=256, imagesize=224):
            self.image = Image.open(image_path).convert("RGB")
            self.transform_img = dataset_library.get_transform(imagesize, resize)
            self.transform_mask = dataset_library.get_transform(imagesize, resize, is_mask=True)

        def __getitem__(self, idx):
            img = self.transform_img(self.image)
            return (img, 0, image_path, "")  # 模拟数据集结构

        def __len__(self):
            return 1

    # 返回修改后的数据加载器生成器
    def get_dataloaders_iter(seed):
        dataset = SingleImageDataset(image_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        dataloader.name = "single_image"
        yield {"testing": dataloader}

    return ("get_dataloaders_iter", [get_dataloaders_iter, 1])


# 修改后的run函数（关键修改部分）
@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images, **kwargs):
    # ... [原有设备初始化代码不变] ...

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        # ... [原有模型加载代码不变] ...

        # 修改后的预测和可视化逻辑
        if "single_image" in dataloaders["testing"].name:
            # 单图推理不需要评估指标
            scores = np.array(aggregator["scores"]).mean(axis=0)
            segmentations = np.array(aggregator["segmentations"]).mean(axis=0)

            # 反标准化预处理
            image = dataloaders["testing"].dataset[0][0].numpy()
            in_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            in_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image = np.clip((image * in_std + in_mean) * 255, 0, 255).transpose(1, 2, 0).astype(np.uint8)

            # 可视化
            if kwargs.get("show_result"):
                plt.figure(figsize=(15, 5))

                # 原始图像
                plt.subplot(1, 3, 1)
                plt.title("Input Image")
                plt.imshow(image)
                plt.axis("off")

                # 异常热力图
                plt.subplot(1, 3, 2)
                plt.title("Anomaly Heatmap")
                plt.imshow(segmentations[0], cmap="jet")
                plt.axis("off")

                # 叠加显示
                plt.subplot(1, 3, 3)
                plt.title("Overlay")
                plt.imshow(image)
                plt.imshow(segmentations[0], cmap="jet", alpha=0.5)
                plt.axis("off")

                plt.tight_layout()
                plt.show()
        else:
    # 原有批量处理逻辑不变
    # ... [原有评估指标计算代码] ...

    # ... [其余原有代码不变] ...

# 其余原有代码保持不变...