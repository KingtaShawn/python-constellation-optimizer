# Python Constellation Optimizer

一个基于深度学习和Sionna框架的星座图优化工具库，用于通信系统中的调制方案设计和性能提升。

## 项目概述

本项目提供了一系列用于通信系统星座图设计、优化和分析的Python工具。通过结合深度学习技术和Sionna通信仿真框架，我们可以设计出比传统QAM等调制方式性能更优的星座图，特别是在高信噪比和衰落信道条件下。

主要功能包括：
- 可微分通信系统的实现
- 星座图的自动优化（基于梯度下降）
- 不同优化策略的星座图设计（功率分配、旋转、交错等）
- 星座图性能分析和可视化
- 信道极化现象模拟
- 高级通信系统性能比较

## 目录结构

项目包含以下主要文件：

```
├── sionna_basis.py              # Sionna框架基础使用示例
├── channel_polarization.py      # 信道极化现象模拟
├── differentiable_Communication_System.py  # 可微分通信系统实现
├── fixed_advanced_constellation.py         # 高级星座图优化实现
├── fixed_constellation_optimizer.py        # 基础星座图优化器
├── constellation_analysis.py    # 星座图分析工具
├── High_level_simulation.py     # 高级通信系统模拟
├── upload_images.py             # 图像上传和管理工具
├── README.md                    # 项目说明文档
├── requirements.txt             # 项目依赖
```

## 安装说明

### 前提条件
- Python 3.7+ 
- TensorFlow 2.8+ 
- NumPy, Matplotlib, PIL等基础库

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用指南

### 1. 基础Sionna使用

`ionna_basis.py`提供了Sionna框架的基础使用示例，包括环境配置、QPSK星座图定义、映射器/解映射器实现、AWGN信道模型和LLR计算等功能。

```bash
python sionna_basis.py
```

### 2. 星座图优化

#### 基础优化
使用`fixed_constellation_optimizer.py`进行基础的星座图优化：

```bash
python fixed_constellation_optimizer.py
```

#### 高级优化
使用`fixed_advanced_constellation.py`进行更复杂的星座图优化，包括距离正则化、课程学习等高级技术：

```bash
python fixed_advanced_constellation.py
```

### 3. 星座图分析

使用`constellation_analysis.py`分析和比较不同星座图的性能：

```bash
python constellation_analysis.py
```

### 4. 通信系统模拟

使用`High_level_simulation.py`进行高级通信系统性能模拟和比较：

```bash
python High_level_simulation.py
```

### 5. 信道极化模拟

使用`channel_polarization.py`模拟和可视化信道极化现象：

```bash
python channel_polarization.py
```

## 项目特点

1. **基于深度学习的优化**：利用自动微分技术对星座图进行端到端优化

2. **多种优化策略**：支持功率分配、旋转、交错等多种星座图优化策略

3. **性能分析工具**：提供完整的星座图性能评估和可视化功能

4. **Sionna框架集成**：与NVIDIA的Sionna通信仿真框架无缝集成

5. **灵活可扩展**：易于扩展新的优化算法和评估指标

## 技术细节

### 星座图优化方法

项目实现了多种星座图优化方法：

- **梯度下降优化**：通过最小化误比特率或二进制交叉熵损失来优化星座点位置

- **距离正则化**：在优化过程中加入距离正则项，确保星座点之间保持足够的最小距离

- **课程学习**：通过逐步增加训练难度（如从低信噪比到高信噪比）来提高优化效果

- **自适应优化**：根据信道条件和系统要求自适应调整优化策略

### 评估指标

项目使用以下指标评估星座图性能：

- 误比特率(BER)
- 误符号率(SER)
- 星座点能量分布
- 星座点最小距离
- 星座图轮廓和对称性

## 应用场景

本项目适用于以下场景：

- 无线通信系统设计
- 卫星通信链路优化
- 衰落信道条件下的调制方案选择
- 高容量通信系统开发
- 通信理论研究和教学

## 注意事项

1. 部分脚本需要GPU支持才能获得最佳性能

2. 运行时间取决于参数设置，复杂的优化可能需要较长时间

3. 生成的图像文件默认保存在当前目录

## 未来工作

1. 添加更多类型的衰落信道模型

2. 集成更多先进的深度学习优化算法

3. 支持多天线系统(MIMO)的星座图优化

4. 开发Web界面便于交互操作

5. 添加更多星座图评估指标和可视化方法

## 许可证

本项目采用MIT许可证。详情请查看LICENSE文件。

## 联系方式

如有任何问题或建议，请联系项目维护者。