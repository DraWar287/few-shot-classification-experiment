"""
小样本分类（Few-shot classification）推理脚本。

任务设定（与 task.md 对齐）：
- 每个 task 下包含 support/ 与 query/：
  - support/：按类别分文件夹，每类 5 张图（N-way 5-shot）
  - query/：20 张图，类别未知
- 同一 task 的 support 与 query 共享同一标签空间
- 目标：为 query 内每张图预测标签（输出为 CSV：img_name,label）

实现思路（原型网络 / Prototypical Networks 的经典推理方式）：
1) 用 backbone 抽取图像特征（embedding）
2) 对 support 中每个类别的 5 个 embedding 求均值，得到该类“原型” prototype
3) 对 query embedding 与所有 prototypes 做相似度（这里使用余弦：先 L2 normalize，再点积）
4) 取相似度最大的类别作为预测

性能策略：
- 使用 DataLoader 批量推理 + GPU AMP（可选）减少 Python 循环与算子开销
- 在一个 task 内全程走内存，不做特征落盘
- 余弦相似度用一次矩阵乘法完成：scores = query_feats @ prototypes.T
"""

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import const


class ImagePathDataset(Dataset):
    """
    只负责“从磁盘读取图片 + 应用 transform”，并把 meta 原样返回。

    items 的结构约定：
    - [(img_path, meta), ...]
    meta 可以是：
    - int（support 场景：类别 id）
    - str（query 场景：图片文件名，用于生成最终输出）
    """

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, meta = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), meta


class EfficientNetB2Extractor(nn.Module):
    """
    EfficientNet-B2 特征抽取器（依赖 efficientnet_pytorch）。

    输出为二维张量 [B, C]：
    - EfficientNet.extract_features 输出 [B, C, H, W]
    - 做全局平均池化 mean(H,W) 得到 [B, C]
    """

    def __init__(self, weights_path):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        self.net = EfficientNet.from_name(const.EFFICIENTNET_PYTORCH_NAME)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.net.load_state_dict(state)

    def forward(self, x):
        x = self.net.extract_features(x)
        x = x.mean(dim=(2, 3))
        return x


def _build_transform():
    """
    推理预处理：
    - Resize 到固定尺寸（匹配大多数 backbone 的输入）
    - ToTensor + Normalize（ImageNet 统计量）
    """

    return transforms.Compose(
        [
            transforms.Resize((const.IMAGE_SIZE, const.IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD),
        ]
    )

def _build_hflip_transform():
    """
    最简 TTA：水平翻转视角。
    - 通过 p=1.0 的 RandomHorizontalFlip 实现“确定性翻转”
    - 和 base transform 保持同样的 resize/normalize
    """

    b = float(getattr(const, "TTA_JITTER_BRIGHTNESS", 0.0))
    c = float(getattr(const, "TTA_JITTER_CONTRAST", 0.0))
    color_jitter = transforms.ColorJitter(brightness=b, contrast=c) if (b > 0.0 or c > 0.0) else None

    return transforms.Compose(
        [
            transforms.Resize((const.IMAGE_SIZE, const.IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=1.0),
            *( [color_jitter] if color_jitter is not None else [] ),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD),
        ]
    )


def _build_model(device):
    """
    构建 backbone，并设置推理相关的加速选项。

    选择逻辑：
    - 若安装了 efficientnet_pytorch：优先用 EfficientNet-B2（可加载本地权重）
    - 否则：用 torchvision 的 ResNet（不下载预训练权重，避免联网依赖）
    """

    torch.backends.cudnn.benchmark = bool(const.CUDNN_BENCHMARK)
    if const.MATMUL_PRECISION:
        torch.set_float32_matmul_precision(const.MATMUL_PRECISION)

    if importlib.util.find_spec("efficientnet_pytorch") is not None:
        weights_path = os.path.join(const.PROJECT_DIR, const.EFFICIENTNET_WEIGHTS_FILENAME)
        model = EfficientNetB2Extractor(weights_path)
    else:
        if const.TORCHVISION_FALLBACK == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Identity()
        elif const.TORCHVISION_FALLBACK == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Identity()
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Identity()

    model.eval().to(device)
    if const.CHANNELS_LAST:
        # channels_last 对 GPU 的卷积常见更友好；输入张量也会配套转换。
        model = model.to(memory_format=torch.channels_last)
    return model


def _make_loader(dataset, batch_size, device):
    """
    统一构建 DataLoader，集中控制 num_workers / pin_memory 等参数。
    """

    num_workers = int(const.NUM_WORKERS)
    pin_memory = bool(const.PIN_MEMORY) and device.type == "cuda"
    persistent_workers = bool(const.PERSISTENT_WORKERS) and num_workers > 0
    prefetch_factor = int(const.PREFETCH_FACTOR) if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _extract_features(items, batch_size, model, device, transform):
    """
    对一组图片路径 items 批量抽取特征。

    返回：
    - feats: [N, D] 的 float 张量（已 L2 normalize）
    - metas: 与输入 items 对齐的 meta（support 为类别 id 的 Tensor；query 为文件名 list）

    关键点：
    - 使用 torch.inference_mode() 关闭 autograd，减少开销
    - 在 CUDA 上可开启 AMP：更快且更省显存
    - 统一做 F.normalize，使“点积”直接等价于“余弦相似度”
    """

    dataset = ImagePathDataset(items, transform)
    loader = _make_loader(dataset, batch_size=batch_size, device=device)

    feats = []
    metas = []
    autocast_enabled = device.type == "cuda" and bool(const.USE_AMP)
    autocast_dtype = torch.float16
    with torch.inference_mode():
        for x, meta in loader:
            x = x.to(device, non_blocking=True)
            if const.CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                f = model(x)
                f = F.normalize(f, dim=1)
            feats.append(f)
            metas.append(meta)

    feats = torch.cat(feats, dim=0)
    if isinstance(metas[0], torch.Tensor):
        # meta 为 int 时，DataLoader 会自动把它们堆成 Tensor。
        metas = torch.cat(metas, dim=0)
    else:
        # meta 为 str 时，DataLoader 会返回 list[str]，这里将其扁平化。
        flat = []
        for m in metas:
            flat.extend(list(m))
        metas = flat
    return feats, metas

def _extract_features_with_hflip_tta(items, batch_size, model, device, base_transform, hflip_transform, enable_hflip):
    """
    最简 TTA：同一批图片跑两次（原图 + 水平翻转），对特征取平均。
    """

    feats, metas = _extract_features(items, batch_size=batch_size, model=model, device=device, transform=base_transform)
    if not bool(const.USE_TTA) or not bool(enable_hflip):
        return feats, metas
    feats_flip, _ = _extract_features(items, batch_size=batch_size, model=model, device=device, transform=hflip_transform)
    return (feats + feats_flip) * 0.5, metas


def _list_images(dir_path):
    """
    列出 dir_path 下的图片文件名（不含路径），并排序保证输出稳定。
    """

    names = os.listdir(dir_path)
    names.sort()
    out = []
    for n in names:
        if n.lower().endswith(const.IMAGE_EXTS):
            out.append(n)
    return out


def _predict_one_task(task_dir, model, device, base_transform, hflip_transform):
    """
    对单个 task 进行预测，返回若干行 CSV 记录（不含表头）。

    具体流程：
    1) 读取 support：按类别文件夹枚举图片，构造 (img_path, class_id)
    2) 抽取 support 特征，按 class_id 聚合求均值 -> prototypes
    3) 读取 query：枚举图片，构造 (img_path, img_name)
    4) 抽取 query 特征，scores = query @ prototypes.T，argmax 得到预测类别
    5) 输出行格式："{img_name},{predicted_class_name}"
    """

    support_dir = os.path.join(task_dir, const.SUPPORT_DIRNAME)
    query_dir = os.path.join(task_dir, const.QUERY_DIRNAME)

    class_names = [d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))]
    class_names.sort()

    support_items = []
    support_class_ids = []
    for class_id, class_name in enumerate(class_names):
        class_dir = os.path.join(support_dir, class_name)
        for img_name in _list_images(class_dir):
            support_items.append((os.path.join(class_dir, img_name), class_id))
            support_class_ids.append(class_id)

    support_feats, support_class_ids = _extract_features_with_hflip_tta(
        support_items,
        batch_size=const.SUPPORT_BATCH_SIZE,
        model=model,
        device=device,
        base_transform=base_transform,
        hflip_transform=hflip_transform,
        enable_hflip=const.TTA_SUPPORT_HFLIP,
    )

    num_classes = len(class_names)
    feat_dim = support_feats.shape[1]

    # prototypes[k] = 类 k 的所有 support 特征求和；counts[k] = 类 k 的样本数
    # 最终 prototypes[k] / counts[k] 即为均值原型。
    prototypes = torch.zeros((num_classes, feat_dim), device=device, dtype=support_feats.dtype)
    class_ids = support_class_ids.to(device=device, non_blocking=True, dtype=torch.long)
    prototypes.index_add_(0, class_ids, support_feats)
    counts = torch.bincount(class_ids, minlength=num_classes).clamp_min(1).to(device=device, dtype=support_feats.dtype)
    prototypes = prototypes / counts.unsqueeze(1)
    prototypes = F.normalize(prototypes, dim=1)

    query_items = []
    for img_name in _list_images(query_dir):
        query_items.append((os.path.join(query_dir, img_name), img_name))

    query_feats, query_names = _extract_features_with_hflip_tta(
        query_items,
        batch_size=const.QUERY_BATCH_SIZE,
        model=model,
        device=device,
        base_transform=base_transform,
        hflip_transform=hflip_transform,
        enable_hflip=const.TTA_QUERY_HFLIP,
    )

    # 由于 query_feats 与 prototypes 都已 L2 normalize：
    # query_feats @ prototypes.T 等价于余弦相似度矩阵 [num_query, num_classes]
    scores = query_feats @ prototypes.t()
    pred_ids = scores.argmax(dim=1).tolist()
    return [f"{query_names[i]},{class_names[pred_ids[i]]}" for i in range(len(query_names))]


def main(to_pred_dir, result_save_path):
    """
    评测入口：
    - 输入：to_pred_dir（包含 testA 子目录）与 result_save_path（输出文件路径）
    - 输出：result_save_path 写入 CSV（表头 + 每张 query 的预测）
    """

    device = torch.device("cuda" if const.USE_CUDA and torch.cuda.is_available() else "cpu")
    model = _build_model(device)
    base_transform = _build_transform()
    hflip_transform = _build_hflip_transform()

    root = os.path.join(os.path.abspath(to_pred_dir), const.TEST_SUBDIR)
    tasks = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    tasks.sort()

    res = [const.OUTPUT_HEADER]
    for task_name in tasks:
        task_dir = os.path.join(root, task_name)
        res.extend(
            _predict_one_task(
                task_dir, model=model, device=device, base_transform=base_transform, hflip_transform=hflip_transform
            )
        )

    with open(result_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(res))


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)
