import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

# 自定义数据集类名（根据您的实际数据集修改）
_CUSTOM_CLASSNAMES = [
    "bupi",
    # 添加您的自定义类别
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for custom datasets without ground truth masks."""

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the custom data folder.
            classname: [str or None]. Name of class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available classes.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CUSTOM_CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        nearest_interp = (
            transforms.InterpolationMode.NEAREST
            if hasattr(transforms, "InterpolationMode")
            else PIL.Image.NEAREST
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=nearest_interp),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)

        self.tile_suffixes = ["_left", "_right"]
        self.tiles_per_image = len(self.tile_suffixes)

    def _compute_tile_bboxes(self, width, height):
        base_width = width // self.tiles_per_image
        remainder = width % self.tiles_per_image
        bboxes = []
        left = 0
        for index in range(self.tiles_per_image):
            extra = 1 if index < remainder else 0
            right = left + base_width + extra
            if index == self.tiles_per_image - 1:
                right = width
            bboxes.append((left, 0, right, height))
            left = right
        return bboxes

    def split_pil_image(self, pil_image):
        width, height = pil_image.size
        return [pil_image.crop(bbox) for bbox in self._compute_tile_bboxes(width, height)]

    def _get_tile_suffix(self, slice_idx):
        if 0 <= slice_idx < len(self.tile_suffixes):
            return self.tile_suffixes[slice_idx]
        return f"_tile{slice_idx}"

    def __getitem__(self, idx):
        orig_idx = idx // self.tiles_per_image
        slice_idx = idx % self.tiles_per_image

        classname, anomaly, image_path, mask_path = self.data_to_iterate[orig_idx]
        full_image = PIL.Image.open(image_path).convert("RGB")
        tiles = self.split_pil_image(full_image)
        tile = tiles[slice_idx]
        image = self.transform_img(tile)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            full_mask = PIL.Image.open(mask_path)
            mask_tiles = self.split_pil_image(full_mask)
            mask = self.transform_mask(mask_tiles[slice_idx])
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        slice_suffix = self._get_tile_suffix(slice_idx)
        image_name = "/".join(image_path.split("/")[-4:]) + slice_suffix

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate) * self.tiles_per_image

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)

            # 检查路径是否存在
            if not os.path.exists(classpath):
                print(f"警告: 路径不存在 {classpath}")
                continue

            anomaly_types = os.listdir(classpath)
            imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)

                # 跳过非目录的文件
                if not os.path.isdir(anomaly_path):
                    continue

                anomaly_files = sorted([
                    f for f in os.listdir(anomaly_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])

                # 跳过空目录
                if not anomaly_files:
                    print(f"警告: 目录为空 {anomaly_path}")
                    continue

                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, f) for f in anomaly_files
                ]

                # 训练/验证分割
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                # 为每个图像创建数据项 (classname, anomaly, image_path, None)
                for image_path in imgpaths_per_class[classname][anomaly]:
                    data_to_iterate.append([classname, anomaly, image_path, None])

        return imgpaths_per_class, data_to_iterate
