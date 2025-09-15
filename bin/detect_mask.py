import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD


def load_model(model_dir, device):
    nn_method = FaissNN(False, 4)
    model = PatchCore(device)
    model.load_from_path(model_dir, device, nn_method)
    model.eval()
    return model

# 366,320
# 256,224
def get_transform(resize=366, imagesize=320):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def split_and_process_image(img_path, model, transform, device):
    """处理长图：切分、检测、拼接（问题区域为白色）"""
    # 1. 加载原始图像
    full_img = Image.open(img_path).convert("RGB")
    width, height = full_img.size

    # 2. 切分为左右两部分
    left_img = full_img.crop((0, 0, 1024, 1024))
    right_img = full_img.crop((1024, 0, 2048, 1024))

    # 3. 处理左半部分
    left_tensor = transform(left_img).unsqueeze(0).to(device)
    _, left_masks = model.predict(left_tensor)

    # 转换为0-255的异常掩码（问题区域为白色）
    if isinstance(left_masks[0], torch.Tensor):
        left_mask = (left_masks[0].cpu().numpy())
    else:
        left_mask = left_masks[0]

    # 4. 处理右半部分
    right_tensor = transform(right_img).unsqueeze(0).to(device)
    _, right_masks = model.predict(right_tensor)

    if isinstance(right_masks[0], torch.Tensor):
        right_mask = right_masks[0].cpu().numpy()
    else:
        right_mask = right_masks[0]

    # 5. 创建全尺寸掩码（黑色背景）
    full_mask = np.zeros((height, width), dtype=np.uint8)

    # 6. 缩放掩码到原始切片尺寸
    left_mask_resized = np.array(Image.fromarray(left_mask).resize(
        (1024, 1024), Image.BILINEAR
    ))
    right_mask_resized = np.array(Image.fromarray(right_mask).resize(
        (1024, 1024), Image.BILINEAR
    ))

    # 7. 拼接结果（问题区域为白色）
    full_mask[:, :1024] = left_mask_resized
    full_mask[:, 1024:] = right_mask_resized

    return full_mask


def infer_and_save(model, img_path, out_path, transform, device):
    # 打开图像并获取尺寸
    img = Image.open(img_path)
    width, height = img.size

    # 判断是否为正方形图像（长宽相等）
    if width == height:
        # 正方形图像处理（问题区域为白色）
        img = img.convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        _, masks = model.predict(img_tensor)

        # 转换为0-255的异常掩码（问题区域为白色）
        if isinstance(masks[0], torch.Tensor):
            mask = (masks[0].cpu().numpy()).astype(np.uint8)
        else:
            mask = (masks[0]).astype(np.uint8)

        full_mask = np.zeros((height, width), dtype=np.uint8)

        # 缩放掩码到原始尺寸
        mask_resized = np.array(Image.fromarray(mask).resize(
            (width, height), Image.BILINEAR
        ))

        full_mask = mask_resized

        # 保存掩码（问题区域为白色）
        Image.fromarray((full_mask * 255).astype(np.uint8)).save(out_path)

    else:
        # 长方形图像处理（问题区域为白色）
        mask = split_and_process_image(img_path, model, transform, device)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(out_path)


def main(model_dir, input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, device)
    transform = get_transform()

    if os.path.isfile(input_path):
        out_path = f"mask_{os.path.splitext(os.path.basename(input_path))[0]}.png"
        infer_and_save(model, input_path, out_path, transform, device)
        print(f"保存掩码到: {out_path}")
    else:
        folder_name = os.path.basename(os.path.normpath(input_path))
        out_dir = f"mask_{folder_name}"
        os.makedirs(out_dir, exist_ok=True)
        processed = 0

        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(input_path, fname)
                out_path = os.path.join(out_dir, f"mask_{os.path.splitext(fname)[0]}.png")

                try:
                    infer_and_save(model, img_path, out_path, transform, device)
                    processed += 1
                    print(f"处理完成: {fname} -> {os.path.basename(out_path)}")
                except Exception as e:
                    print(f"处理失败 {fname}: {str(e)}")

        print(f"完成! 共处理 {processed} 张图像")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python detect_mask.py <模型目录> <输入路径>")
        print("输入路径可以是单张图像或包含图像的文件夹")
        sys.exit(1)

    model_dir = sys.argv[1]
    input_path = sys.argv[2]
    main(model_dir, input_path)