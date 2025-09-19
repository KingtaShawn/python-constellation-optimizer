import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

# 抑制所有与字体相关的UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

# 简化字体设置，使用更基础的配置
matplotlib.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.unicode_minus': False  # 解决负号显示问题
})

# 使用系统默认的无衬线字体，避免指定可能不存在的中文字体
# plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]

# 与MATLAB代码对应的Python实现
index = 10
n = 2 ** np.arange(1, index + 1)  # 创建2^1到2^10的数组

# 创建一个足够大的矩阵来存储所有计算结果
max_size = n[-1]  # 2^10 = 1024
W = np.zeros((max_size + 1, max_size + 1))  # 注意这里创建的是1025x1025矩阵，以便使用1-based索引

# 设置初始值 (使用1-based索引以匹配MATLAB)
W[1, 1] = 0.5  # 直接对应MATLAB中的W(1,1) = 0.5

# 迭代计算所有信道的容量 - 完全按照MATLAB的逻辑
for i in n:
    i_int = int(i)
    half_i = i_int // 2
    
    # 按照MATLAB的逻辑，直接使用数值作为索引
    for j in range(1, half_i + 1):
        # 坏信道和好信道的计算
        bad_channel = W[half_i, j] ** 2
        good_channel = 2 * W[half_i, j] - W[half_i, j] ** 2
        
        # 存储计算结果 - 直接使用i作为行索引
        W[i_int, 2 * j - 1] = bad_channel  # 坏信道
        W[i_int, 2 * j] = good_channel  # 好信道

# 绘制信道极化现象（使用最后一行的数据，即2^10=1024个信道）
plt.figure(figsize=(10, 6))
plt.scatter(range(1, 1025), W[1024, 1:1025], s=10, c='r', marker='.')  # 使用1-based索引
plt.axis([0, 1024, 0, 1])
plt.xlabel('信道序号i')
plt.ylabel('对称信道容量I(W)')
plt.title('信道极化现象')
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('channel_polarization.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

print("信道极化模拟完成，图像已保存为'channel_polarization.png'")