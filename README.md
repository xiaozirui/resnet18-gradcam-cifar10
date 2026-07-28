# resnet18-gradcam-cifar10
PyTorch ResNet-18 implementation with Grad-CAM visualization on CIFAR-10

# ResNet-18 From Scratch & Grad-CAM Visualization on CIFAR-10

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

本仓库提供了一个基于 **PyTorch 从零手写（From Scratch）** 的 ResNet-18 图像分类网络实现，并在 **CIFAR-10** 数据集上完成了训练。项目包含完整的模块化代码结构、训练/验证可视化曲线，以及基于 **Grad-CAM** 的可解释性注意力热力图分析。

---

## 🌟 项目亮点 (Key Features)

* **纯手写 ResNet-18 架构**：不依赖 `torchvision.models` 预训练库，从基础 `BasicBlock` 到残差 Stage 纯手撕实现。
* **CIFAR-10 适配优化**：针对 $32 \times 32$ 小分辨率输入优化 Stem 层结构（采用 $3 \times 3$ 卷积，移除初始 MaxPool），防止低高级语义细节过早丢失。
* **模型可解释性 (XAI)**：集成 Grad-CAM 算法，可视化网络在分类预测时的关注区域。
* **模块化工程结构**：严格遵守标准 PyTorch 工业级工程目录组织。

---

## 🛠️ 项目目录结构 (Project Structure)

```text
resnet18-gradcam-cifar10/
├── src/
│   ├── model.py          # ResNet-18 模型搭建与单元测试
│   ├── dataset.py        # CIFAR-10 数据加载与数据增强 (Day 2)
│   ├── utils.py          # 训练指标统计与绘图工具 (Day 2)
│   └── gradcam.py        # Grad-CAM 热力图提取算法 (Day 3)
├── train.py              # 模型训练与验证主入口 (Day 2)
├── visualize.py          # Grad-CAM 热力图可视化脚本 (Day 3)
├── requirements.txt      # 项目依赖包
└── README.md             # 项目说明文档
```

---

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
git clone [https://github.com/xiaozirui/resnet18-gradcam-cifar10.git](https://github.com/xiaozirui/resnet18-gradcam-cifar10.git)
cd resnet18-gradcam-cifar10

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型结构测试

单独运行模型脚本以验证维度流转是否正常：

```bash
python src/model.py
```

---

## 📊 实验结果与可视化 (Results & Grad-CAM)

*(🚧 训练曲线与 Grad-CAM 可视化热力图将在 Day 2 & Day 3 补充)*

---

## 📄 开源协议 (License)

本项目基于 [MIT License](LICENSE) 开源。