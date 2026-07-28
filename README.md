# ResNet-18 From Scratch & Grad-CAM Visualization on CIFAR-10

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

本仓库提供了一个基于 **PyTorch 从零手写（From Scratch）** 的 ResNet-18 图像分类网络完整实现，并在 **CIFAR-10** 数据集上完成了训练与测试。项目包含标准的模块化代码结构、训练/验证可视化指标曲线，以及基于 **Grad-CAM** 算法的可解释性注意力热力图分析。

---

## 🌟 项目亮点 (Key Features)

* **纯手写 ResNet-18 架构**：拒绝 `torchvision.models` 预训练黑盒，从基础 `BasicBlock` 残差块到多 Stage 网络实现。
* **小尺寸输入针对性优化**：针对 CIFAR-10 数据集 $32 \times 32$ 的低分辨率输入，改写 Stem 层（$3 \times 3$ 卷积、Stride=1、移除 Initial MaxPool），防止浅层空间特征过早丢失。
* **模型可解释性 (XAI)**：自研 Grad-CAM 热力图提取算法，深入网络内部提取特定层特征图与梯度映射，直观可视化模型的决策关注域。
* **工业级工程标准**：遵守标准 PyTorch 模块化组织架构，包含单元测试 (Unit Test)、数据集自动下载与增强、检查点自动保存及断点恢复机制。

---

## 📅 项目开发路线 (Roadmap)

- [x] **Day 1**: 手撕 ResNet-18 架构与维度流转单元测试
- [x] **Day 2**: 构建 CIFAR-10 数据增强流水线、后勤评估工具与主训练循环 (`train.py`)
- [ ] **Day 3**: 实现 Grad-CAM 热力图算法与可视化脚本 (`visualize.py`)

---

## 🛠️ 项目目录结构 (Project Structure)

```text
resnet18-gradcam-cifar10/
├── checkpoints/          # 模型权重保存目录 (*.pth)
├── data/                 # CIFAR-10 数据集下载与存储目录
├── outputs/              # 训练曲线图 (loss_acc.png) 与 Grad-CAM 热力图输出目录
├── src/
│   ├── model.py          # ResNet-18 模型搭建与维度单元测试
│   ├── dataset.py        # CIFAR-10 数据加载与增强流水线
│   ├── utils.py          # 训练指标统计与 Loss/Acc 曲线绘制工具
│   └── gradcam.py        # Grad-CAM 热力图提取核心算法 (Day 3)
├── train.py              # 模型训练与验证主循环脚本
├── visualize.py          # Grad-CAM 热力图可视化脚本 (Day 3)
├── .gitignore            # Git 忽略文件配置
├── requirements.txt      # 项目环境依赖包
├── LICENSE               # MIT 开源协议
└── README.md             # 项目说明文档
```

## 📐 模型架构与设计细节 (Architecture & Design)

针对 CIFAR-10 的 $32 \times 32$ 图像输入，本项目对经典 ResNet-18 架构进行了微调：

| 模块名称 (Stage) | 输入 Shape | 输出 Shape | 关键设计与参数 |
| :--- | :--- | :--- | :--- |
| **Stem Layer** | $(3, 32, 32)$ | $(64, 32, 32)$ | $3 \times 3$ Conv, Stride=1, Padding=1 (保留空间分辨率) |
| **Layer 1** | $(64, 32, 32)$ | $(64, 32, 32)$ | 2 $\times$ BasicBlock, Stride=1 |
| **Layer 2** | $(64, 32, 32)$ | $(128, 16, 16)$ | 2 $\times$ BasicBlock, 首块 Stride=2 (下采样与通道翻倍) |
| **Layer 3** | $(128, 16, 16)$ | $(256, 8, 8)$ | 2 $\times$ BasicBlock, 首块 Stride=2 (下采样与通道翻倍) |
| **Layer 4** | $(256, 8, 8)$ | $(512, 4, 4)$ | 2 $\times$ BasicBlock, 首块 Stride=2 (下采样与通道翻倍) |
| **Classifier** | $(512, 4, 4)$ | $(10)$ | AdaptiveAvgPool2d((1,1)) $\to$ Flatten $\to$ Linear(512, 10) |

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/xiaozirui/resnet18-gradcam-cifar10.git
cd resnet18-gradcam-cifar10

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型结构测试

单独运行模型脚本以验证维度流转是否正常：

```bash
# 测试模型结构与维度流转
python src/model.py

# 测试数据集加载与增强策略
python src/dataset.py

# 测试评估指标与绘图工具
python src/utils.py
```
运行训练脚本，数据将自动下载至 data/ 目录，训练过程中的最优权重将保存在 checkpoints/best_model.pth，曲线图将自动保存至 outputs/loss_acc.png：
```bash
python train.py
```

---

## 📊 实验结果与可视化 (Results & Grad-CAM)

*(🚧 训练曲线与 Grad-CAM 可视化热力图将在 Day 2 & Day 3 补充)*

---

## 📄 开源协议 (License)

本项目基于 [MIT License](LICENSE) 开源。
