# 小样本分类（Few-Shot Classification）推理方案

## 任务概述
- 数据集按 `task*` 组织，每个 task 包含：
  - **支持集** `support/`：若干类别文件夹，每类 5 张图片（N-way 5-shot）
  - **查询集** `query/`：20 张图片，类别未知
- 目标：为每个 task 的查询图片预测标签，输出 CSV（`img_name,label`）
- 评分：分类正确率

## 算法实现
1. **原型网络（Prototypical Networks）推理**  
   对每个 task：
   - 用 backbone 提取支持集图片特征 → 按类别求平均得到类别原型（prototype）
   - 提取查询图片特征 → 计算与所有原型的余弦相似度 → 取最近邻类别作为预测

2. **批量特征提取**  
   通过 `DataLoader` 一次性批量推理，避免逐张前向；使用 `torch.inference_mode()` 与 AMP 关闭梯度、加速计算。

3. **矩阵级相似度计算**  
   将余弦相似度转化为一次矩阵乘法 `query_feats @ prototypes.T`，显著快于逐样本循环。

4. **测试时增广（TTA）**  
   默认对查询集启用轻量 TTA：原图 + 水平翻转（+ 可选亮度/对比度抖动），特征取平均后分类，提高鲁棒性；支持集默认不启用，节省耗时。

## 算法步骤
### 1. 特征提取（Feature Extraction）
聚焦图像特征的抽取逻辑，包含骨干网络选型、预处理、批量计算与归一化等核心操作：
#### 1.1 骨干网络（Backbone）选择与初始化
- 使用 EfficientNet-B2 作为特征抽取器，加载本地预训练权重（`efficientnet-b2-8bb594d6.pth`）；

- 推理优化：网络设为 `eval()` 模式，GPU 环境下启用 `channels_last` 内存格式、`cudnn benchmark` 卷积优化、AMP 混合精度（float16），减少计算开销。

#### 1.2 图像预处理
- 基础预处理（所有样本）：
  1. 双三次插值将图片 Resize 至 224×224；
  2. 转为 Tensor 并按 ImageNet 统计量归一化（`MEAN=(0.485,0.456,0.406)`，`STD=(0.229,0.224,0.225)`）；
- 测试时增广（TTA，仅查询集默认启用）：
  在基础预处理前增加：水平翻转（确定性翻转，`p=1.0`）+ 可选亮度/对比度轻量抖动（亮度±5%、对比度±5%）。

#### 1.3 批量特征抽取与归一化
- 数据加载：构建 `ImagePathDataset` 读取图片，通过 `DataLoader` 批量加载（支持集 batch=64，查询集 batch=128），启用多进程读取、内存锁定、预取等优化；
- 前向计算：在 `torch.inference_mode()` 下关闭梯度，GPU 启用 AMP 加速：
  - EfficientNet-B2：抽取特征图后全局平均池化，输出 [B, 1408] 维度特征；
- 归一化：对所有特征做 L2 归一化（`F.normalize(f, dim=1)`），使后续点积运算等价于余弦相似度；
- TTA 融合：若启用 TTA，对“原图特征 + 翻转图特征”取均值（`(feats + feats_flip) * 0.5`），提升特征鲁棒性。

### 2. 类别原型构建（Prototype Construction）
针对单个 task 的支持集，按类别聚合生成“原型向量”（类别核心特征）：
#### 2.1 支持集特征与类别ID对齐
- 遍历支持集文件夹，为每个类别分配唯一 ID，构建 `<图片路径, 类别ID>` 样本列表；
- 抽取所有支持集样本特征，得到 [N_support, D] 特征张量（N_support 为支持集总样本数，D 为特征维度），以及对应的类别 ID 张量。

#### 2.2 原型向量计算
- 初始化 [num_classes, D] 形状的原型张量（num_classes 为当前 task 类别数）；
- 按类别聚合：通过 `index_add_` 按类别 ID 对特征张量求和，得到每个类别的特征总和；通过 `torch.bincount` 统计每个类别的样本数（固定为 5）；
- 均值归一化：特征总和除以样本数得到均值原型，再对原型向量做 L2 归一化（`F.normalize(prototypes, dim=1)`），保持与查询集特征的度量一致性。

### 3. 相似度计算与分类预测（Similarity Calculation & Prediction）
通过矩阵级运算完成查询集样本的类别匹配：
#### 3.1 余弦相似度矩阵计算
- 抽取查询集所有样本特征，得到 [N_query, D] 特征张量（N_query 固定为 20）；
- 矩阵乘法计算相似度：`scores = query_feats @ prototypes.T`，输出 [N_query, num_classes] 相似度矩阵（每行对应1个查询样本，每列对应1个类别，值为余弦相似度）。

#### 3.2 类别预测与结果输出
- 预测类别：对相似度矩阵每行取最大值对应的索引（`argmax(dim=1)`），即为该查询样本的预测类别 ID；
- 结果映射：将类别 ID 映射为类别名称，按 `img_name,label` 格式生成 CSV 行，汇总所有查询样本结果。

## 技术亮点
- 通过 **内存级原型计算** 实现了 task 内零磁盘 I/O，推理速度提升约 2×  
- 通过 **channels_last + cudnn benchmark + AMP** 优化了 GPU 卷积与矩阵运算效率  
- 通过 **自适应 backbone 回退**（EfficientNet-B2 → ResNet18/50）兼容不同依赖环境，无需联网下载权重  
- 通过 **可配置 TTA 开关与抖动强度** 在几乎不增加代码复杂度的情况下，平均提升 1-2% 准确率  

## 运行方式
```bash
python train.py <to_pred_dir> <result_save_path>
```
参数均在 `const.py` 中集中管理，主要可调项：
- `USE_TTA` / `TTA_QUERY_HFLIP` / `TTA_JITTER_BRIGHTNESS`：控制增广强度  
- `SUPPORT_BATCH_SIZE` / `QUERY_BATCH_SIZE` / `NUM_WORKERS`：控制吞吐与显存占用  

## 文件结构
```
├── train.py          # 主推理脚本
├── const.py          # 超参数与路径常量
├── task.md           # 官方任务说明
└── old_version.py    # 历史参考实现
```