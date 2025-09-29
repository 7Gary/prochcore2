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
    """
    PyTorch Dataset for custom datasets without ground truth masks.
    """

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

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        # 计算原始图像索引和切片索引
        orig_idx = idx // 2
        slice_idx = idx % 2  # 0=左切片, 1=右切片

        classname, anomaly, image_path, mask_path = self.data_to_iterate[orig_idx]
        full_image = PIL.Image.open(image_path).convert("RGB")

        # 定义两个切片的边界框
        if slice_idx == 0:  # 左切片
            bbox = (0, 0, 1024, 1024)
        else:  # 右切片
            bbox = (1024, 0, 2048, 1024)

        # 裁剪图像
        tile = full_image.crop(bbox)
        image = self.transform_img(tile)

        # 处理掩码
        if self.split == DatasetSplit.TEST and mask_path is not None:
            full_mask = PIL.Image.open(mask_path)
            mask = full_mask.crop(bbox)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        # 添加切片标识到图像名称
        slice_suffix = "_left" if slice_idx == 0 else "_right"
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

        # return {
        #     "image": image,
        #     "mask": mask,
        #     "classname": classname,
        #     "anomaly": anomaly,
        #     "is_anomaly": int(anomaly != "good"),
        #     "image_name": "/".join(image_path.split("/")[-4:]),
        #     "image_path": image_path,
        # }

    def __len__(self):
        return len(self.data_to_iterate) * 2

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