# Python 星座图优化项目 - 快速入门指南

本指南将帮助您快速启动并运行Python星座图优化项目。

## 1. 环境设置

### 1.1 克隆仓库

```bash
# 克隆仓库到本地
https://github.com/KingtaShawn/python-constellation-optimizer.git
cd python-constellation-optimizer
```

### 1.2 创建虚拟环境

```bash
# 创建Python虚拟环境
python -m venv venv_new

# 激活虚拟环境
# Windows系统
env\Scripts\activate
# macOS/Linux系统
source venv_new/bin/activate
```

### 1.3 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 1.4 环境配置

如果您计划使用GPU进行训练，请确保您的系统已正确安装CUDA和cuDNN，并且与您安装的TensorFlow版本兼容。

## 2. 快速开始示例

### 2.1 运行简单的星座图优化

```bash
# 运行简单的星座图优化脚本
python simple_constellation_optimizer.py
```

此脚本实现了基本的星座图优化功能，适合初学者理解项目的核心概念。

### 2.2 运行高级Sionna星座图优化

```bash
# 运行高级Sionna星座图优化脚本
python advanced_sionna_constellation.py
```

此脚本提供了更高级的星座图优化功能，包括多种星座类型支持、距离正则化和课程学习策略。

### 2.3 星座图分析

```bash
# 运行星座图分析脚本
python constellation_analysis.py
```

此脚本可用于分析和可视化不同的星座图性能特征。

## 3. 主要功能模块

### 3.1 星座图优化

- **可训练星座图**：通过梯度下降优化星座点位置
- **距离正则化**：确保优化后的星座图保持足够的最小距离
- **课程学习**：从低信噪比开始，逐渐增加到高信噪比

### 3.2 通信系统仿真

- **端到端系统**：包含映射器、信道和解映射器的完整通信链路
- **多种调制方式**：支持QAM、PSK和自定义星座图
- **AWGN和Rayleigh信道**：支持不同的信道模型

### 3.3 性能评估

- **BER计算**：误比特率评估
- **SER计算**：误符号率评估
- **星座图可视化**：直观展示星座点分布

## 4. 项目文件结构

- `differentiable_Communication_System.py`：可微分通信系统的核心实现
- `fixed_advanced_constellation.py`：高级固定星座图优化实现
- `fixed_constellation_optimizer.py`：基础固定星座图优化器
- `advanced_sionna_constellation.py`：基于Sionna库的高级星座图优化
- `simple_constellation_optimizer.py`：简单星座图优化示例
- `constellation_analysis.py`：星座图分析工具
- `High_level_simulation.py`：高层级系统仿真脚本
- `upload_images.py`：图像上传和管理工具
- `README.md`：项目详细说明文档
- `README_FIX.txt`：环境配置和问题修复指南
- `requirements.txt`：项目依赖列表

## 5. 常见问题解决

### 5.1 Sionna库安装问题

如果遇到Sionna库安装问题，请参考`README_FIX.txt`中的详细指南。

### 5.2 GPU内存不足

如果在训练过程中遇到GPU内存不足的错误，可以尝试：
- 减小批量大小
- 减小星座图大小（例如，从64-QAM改为16-QAM）
- 使用`tf.config.experimental.set_memory_growth`启用内存增长模式

### 5.3 中文显示问题

项目已配置支持中文字体，但如果图表中中文显示异常，请检查您的Matplotlib配置。

## 6. 项目扩展建议

如果您想进一步扩展此项目，可以考虑：

- 实现更多类型的信道模型（如衰落信道、干扰信道等）
- 探索深度学习辅助的解调方法
- 优化高谱效调制方案（如非均匀星座图）
- 实现多输入多输出（MIMO）系统中的星座图优化

## 7. 联系方式

如有任何问题或建议，请通过以下方式联系项目维护者：

- GitHub Issues：在项目仓库中提交问题
- 电子邮件：[caojh2005@163.com](mailto:caojh2005@163.com)

祝您使用愉快！