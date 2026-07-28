# CIFAR-10 上的 ResNet-18 实现与 Grad-CAM 可视化

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 项目简介

本项目使用 PyTorch 实现适用于 CIFAR-10 数据集的 ResNet-18 图像分类模型。模型由基础残差块逐层构建，未调用 `torchvision.models` 中的预定义网络；项目同时提供数据加载、训练、验证与训练指标可视化功能。

网络结构针对 CIFAR-10 的 $32 \times 32$ 输入进行了调整：Stem 层采用 $3 \times 3$、步长为 1 的卷积，并省略初始最大池化层，以减少浅层特征的空间信息损失。

Grad-CAM 可视化功能已列入开发计划，当前版本尚未包含对应实现。

## 功能概览

- 手动实现 ResNet-18 的 `BasicBlock` 与四个残差阶段。
- 自动下载 CIFAR-10 数据集，并提供训练集数据增强与标准化处理。
- 支持训练集与验证集的损失、准确率统计及曲线保存。
- 在验证准确率提升时保存最优模型权重。
- 自动选择 CUDA、Apple MPS 或 CPU 作为运行设备。

## 模型结构

| 模块 | 输入维度 | 输出维度 | 配置 |
| :--- | :--- | :--- | :--- |
| Stem | $(3, 32, 32)$ | $(64, 32, 32)$ | $3 \times 3$ 卷积，stride=1，padding=1 |
| Layer 1 | $(64, 32, 32)$ | $(64, 32, 32)$ | 2 个 `BasicBlock`，stride=1 |
| Layer 2 | $(64, 32, 32)$ | $(128, 16, 16)$ | 2 个 `BasicBlock`，首块 stride=2 |
| Layer 3 | $(128, 16, 16)$ | $(256, 8, 8)$ | 2 个 `BasicBlock`，首块 stride=2 |
| Layer 4 | $(256, 8, 8)$ | $(512, 4, 4)$ | 2 个 `BasicBlock`，首块 stride=2 |
| 分类器 | $(512, 4, 4)$ | $(10)$ | 自适应平均池化、展平与全连接层 |

## 目录结构

```text
resnet18-gradcam-cifar10/
├── src/
│   ├── dataset.py        # CIFAR-10 数据集与数据增强
│   ├── model.py          # ResNet-18 模型定义
│   └── utils.py          # 评估指标与训练曲线绘制
├── train.py              # 训练与验证入口
├── requirements.txt      # Python 依赖
├── LICENSE               # MIT 许可证
└── README.md             # 项目说明
```

运行训练或测试脚本后，项目会按需生成以下目录：

- `data/`：CIFAR-10 数据集文件。
- `checkpoints/`：验证集准确率最佳的模型权重，默认文件名为 `best_model.pth`。
- `outputs/`：训练曲线及测试脚本生成的图像。

## 环境要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本

安装依赖：

```bash
git clone https://github.com/xiaozirui/resnet18-gradcam-cifar10.git
cd resnet18-gradcam-cifar10
pip install -r requirements.txt
```

## 使用说明

### 模块验证

可分别执行以下脚本，检查模型输出维度、数据加载流程与工具函数：

```bash
python src/model.py
python src/dataset.py
python src/utils.py
```

其中，`src/dataset.py` 首次运行时会下载 CIFAR-10 数据集；`src/utils.py` 会在 `outputs/` 目录生成测试图像。

### 训练模型

```bash
python train.py
```

训练脚本当前使用批大小 128、训练 20 个 epoch，并采用带动量和权重衰减的 SGD 优化器及余弦退火学习率调度。训练完成后：

- 最优权重保存至 `checkpoints/best_model.pth`；
- 损失与准确率曲线保存至 `outputs/loss_acc.png`。

## 开发计划

- [x] ResNet-18 网络实现与输出维度验证。
- [x] CIFAR-10 数据管道、训练流程与指标可视化。
- [ ] Grad-CAM 热力图生成与可视化脚本。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
