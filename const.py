"""
本文件集中管理 train.py 的所有可调参数与常量。

说明：
- 这里的默认值偏向“条件较优时尽可能快”（有 GPU / 充足内存 / SSD）。
- 不考虑异常处理时，这些参数在大多数环境下可直接使用；如遇到显存不足、
  DataLoader 卡死等情况，再按注释建议调整即可。
"""

import os

# 项目根目录（即本文件所在目录）。用于定位权重文件等相对路径资源。
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录约定：
# to_pred_dir/
#   testA/
#     task0/
#       support/0/*.png ...
#       query/*.png ...
TEST_SUBDIR = "testA"
SUPPORT_DIRNAME = "support"
QUERY_DIRNAME = "query"

# 允许的图片后缀
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# 推理输入尺寸。越大特征越精细但越慢；一般 224 是速度/效果较均衡的选择。
IMAGE_SIZE = 224
# 常见 ImageNet 归一化参数。与大多数预训练 backbone 的训练分布一致。
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# 测试时增广（Test-Time Augmentation, TTA）
# 最简单且通常有效的方案：对同一张图同时跑“原图”和“水平翻转图”，
# 再对两次特征取平均。这样实现简单、稳定，且只会让推理变慢约 2 倍。
USE_TTA = True

# 建议只对 query 使用翻转 TTA（更省时，且通常带来一点点提升）。
TTA_SUPPORT_HFLIP = False
TTA_QUERY_HFLIP = True

# 亮度/对比度轻量抖动（仅用于 TTA 视角；base view 不抖动）。
# 取值含义同 torchvision.transforms.ColorJitter：
# - brightness=0.05 表示亮度在 [0.95, 1.05] 范围内随机缩放
# - contrast=0.05 表示对比度在 [0.95, 1.05] 范围内随机缩放
TTA_JITTER_BRIGHTNESS = 0.05
TTA_JITTER_CONTRAST = 0.05

# DataLoader 批大小：
# - 支持集：每类 5 张图，N-way 时总量通常不大；batch 可适当大以减少迭代次数。
# - 查询集：固定 20 张图，batch 设置为 128/256 主要是兼顾不同任务与扩展性。
SUPPORT_BATCH_SIZE = 64
QUERY_BATCH_SIZE = 128

# DataLoader 并行读取的进程数。通常设置为 CPU 核数-1，且上限 8。
# 如果在 Windows 上遇到 DataLoader 启动慢/卡住，可将其改为 0（退化为主进程读取）。
NUM_WORKERS = max(0, min(8, (os.cpu_count() or 0) - 1))

# pin_memory 在 GPU 场景通常可提高 Host->Device 传输效率；CPU 推理则无意义。
PIN_MEMORY = True

# persistent_workers 能减少每轮 DataLoader 重建进程的开销，适合多轮/多 task 推理。
PERSISTENT_WORKERS = True

# 预取 batch 数量，通常 2~8。过大可能增加内存占用。
PREFETCH_FACTOR = 4

# 是否优先使用 GPU。即使为 True，也会在无 GPU 时自动回退到 CPU。
USE_CUDA = True

# 自动混合精度（AMP）。GPU 推理通常更快更省显存；若发现精度/稳定性问题可关掉。
USE_AMP = True

# channels_last 内存格式：对部分卷积网络在 GPU 上更快。CPU 上一般无明显收益。
CHANNELS_LAST = True

# cudnn benchmark：输入尺寸固定时可让 cuDNN 选择最快卷积算法，通常能提速。
CUDNN_BENCHMARK = True

# matmul 精度策略（PyTorch 2+）："high"/"medium"/"highest" 或 None。
# 更高的设置通常更快（依赖硬件与 PyTorch 版本实现）。
MATMUL_PRECISION = "high"

# 如果安装了 efficientnet_pytorch，则优先使用其 EfficientNet-B2 结构抽取特征。
# 权重文件按旧版约定放在项目目录内（与 train.py 同级）。
EFFICIENTNET_PYTORCH_NAME = "efficientnet-b2"
EFFICIENTNET_WEIGHTS_FILENAME = "efficientnet-b2-8bb594d6.pth"

# 如果没有 efficientnet_pytorch，则使用 torchvision 的 backbone 作为回退方案。
# 可选："resnet18" / "resnet50"。这里不下载预训练权重，以避免联网依赖。
TORCHVISION_FALLBACK = "resnet18"

# 输出 CSV 的表头，需与评测端读取逻辑一致。
OUTPUT_HEADER = "img_name,label"
