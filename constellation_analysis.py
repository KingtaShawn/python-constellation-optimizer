import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 配置环境变量以使用特定GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # 使用空字符串""表示使用CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志输出

# 导入Sionna库
try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    print("Sionna库未安装，尝试安装...")
    os.system("pip install sionna")
    # 安装后尝试重新导入
    try:
        import sionna as sn
        import sionna.phy
        print("Sionna库安装成功！")
    except ImportError:
        print("Sionna库安装失败，请手动安装后再运行此脚本。")
        sys.exit(1)

# 配置TensorFlow使用GPU内存增长模式
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU {gpus[0].name} 已配置为内存增长模式")
    except RuntimeError as e:
        print(f"配置GPU内存增长模式失败: {e}")

tf.get_logger().setLevel('ERROR')  # 关闭TensorFlow警告日志

# 设置随机种子以确保结果可复现
sn.phy.config.seed = 42
np.random.seed(42)
tf.random.set_seed(42)

# 自定义星座类用于加载预训练的星座点
class CustomConstellation(tf.keras.layers.Layer):
    def __init__(self, num_bits_per_symbol, points=None):
        super(CustomConstellation, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        
        if points is None:
            # 如果没有提供星座点，则使用默认QAM星座
            qam_constellation = sn.phy.mapping.Constellation("qam", num_bits_per_symbol)
            initial_points = tf.stack([tf.math.real(qam_constellation.points),
                                       tf.math.imag(qam_constellation.points)], axis=0)
            self.points = tf.Variable(initial_points, dtype=tf.float32, trainable=False)
        else:
            # 使用提供的星座点
            if isinstance(points, np.ndarray):
                points = tf.convert_to_tensor(points, dtype=tf.float32)
            self.points = tf.Variable(points, dtype=tf.float32, trainable=False)
    
    def call(self, inputs=None):
        # 归一化星座点，保持能量恒定
        normalized_points = self.normalize_constellation(self.points)
        
        # 将星座点转换为复数形式
        complex_points = tf.complex(normalized_points[0], normalized_points[1])
        
        return complex_points
    
    def normalize_constellation(self, points):
        # 计算星座点的平均能量
        power = tf.reduce_mean(tf.square(points[0]) + tf.square(points[1]))
        
        # 归一化星座点，使其平均能量为1
        normalized_points = points / tf.sqrt(power)
        
        return normalized_points

# 从文件加载预训练的星座点
def load_pretrained_constellation(weights_path, num_bits_per_symbol):
    # 创建一个临时模型来加载权重
    temp_model = tf.keras.Sequential([
        CustomConstellation(num_bits_per_symbol)
    ])
    
    # 构建模型
    _ = temp_model(None)
    
    try:
        # 尝试加载权重
        temp_model.load_weights(weights_path)
        print(f"成功加载预训练星座点权重: {weights_path}")
        
        # 获取加载的星座点
        constellation = temp_model.layers[0]
        points = constellation.points.numpy()
        
        return points
    except Exception as e:
        print(f"加载权重失败: {e}")
        print("使用默认QAM星座点")
        return None

# 计算星座点的基本统计特征
def calculate_constellation_stats(constellation_points):
    # 将复数星座点转换为实部和虚部
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    # 计算基本统计特征
    stats = {
        "mean_real": np.mean(real_parts),
        "mean_imag": np.mean(imag_parts),
        "std_real": np.std(real_parts),
        "std_imag": np.std(imag_parts),
        "max_amplitude": np.max(np.sqrt(real_parts**2 + imag_parts**2)),
        "min_amplitude": np.min(np.sqrt(real_parts**2 + imag_parts**2)),
        "avg_amplitude": np.mean(np.sqrt(real_parts**2 + imag_parts**2))
    }
    
    return stats

# 计算星座点之间的最小距离
def calculate_min_distance(constellation_points):
    # 将复数星座点转换为实部和虚部的数组
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    # 创建所有点对的索引
    num_points = len(real_parts)
    min_dist = float('inf')
    
    # 遍历所有点对，计算最小距离
    for i in range(num_points):
        for j in range(i + 1, num_points):
            real_diff = real_parts[i] - real_parts[j]
            imag_diff = imag_parts[i] - imag_parts[j]
            distance = np.sqrt(real_diff**2 + imag_diff**2)
            
            if distance < min_dist:
                min_dist = distance
    
    return min_dist

# 分析星座点分布特征
def analyze_constellation_distribution(constellation_points):
    # 将复数星座点转换为实部和虚部
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    # 计算星座点的分布特征
    # 检查是否是规则网格分布（如QAM）
    # 提取唯一的实部和虚部值
    unique_real = np.unique(np.round(real_parts, decimals=4))
    unique_imag = np.unique(np.round(imag_parts, decimals=4))
    
    # 计算实部和虚部的差异
    real_diff = np.diff(unique_real)
    imag_diff = np.diff(unique_imag)
    
    # 检查是否是规则网格
    is_regular_grid = False
    if len(real_diff) > 0 and len(imag_diff) > 0:
        is_regular_grid_real = np.allclose(real_diff, real_diff[0], rtol=1e-2)
        is_regular_grid_imag = np.allclose(imag_diff, imag_diff[0], rtol=1e-2)
        is_regular_grid = is_regular_grid_real and is_regular_grid_imag
    
    # 计算星座点的分布熵（衡量分布的均匀性）
    # 将复平面划分为网格
    grid_size = 100
    hist, _, _ = np.histogram2d(real_parts, imag_parts, bins=grid_size, range=[[-2, 2], [-2, 2]])
    
    # 计算熵
    hist = hist / np.sum(hist)  # 归一化
    hist = hist[hist > 0]  # 只考虑非零概率
    entropy = -np.sum(hist * np.log2(hist))
    
    # 计算最大可能熵（均匀分布）
    max_entropy = np.log2(len(constellation_points))
    
    # 计算熵效率
    entropy_efficiency = entropy / max_entropy
    
    distribution_analysis = {
        "is_regular_grid": is_regular_grid,
        "num_unique_real": len(unique_real),
        "num_unique_imag": len(unique_imag),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_efficiency": entropy_efficiency
    }
    
    return distribution_analysis

# 绘制星座图热力图（显示星座点分布密度）
def plot_constellation_heatmap(constellation_points, title="星座点分布热力图"):
    # 将复数星座点转换为实部和虚部
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    plt.figure(figsize=(10, 8))
    
    # 绘制二维直方图（热力图）
    h, xedges, yedges, image = plt.hist2d(real_parts, imag_parts, bins=100,
                                         range=[[-2, 2], [-2, 2]],
                                         cmap=cm.jet, norm=cm.colors.LogNorm())
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('点密度')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # 保存图表
    plt.savefig('constellation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("星座点分布热力图已保存为 'constellation_heatmap.png'")

# 绘制星座点的星座图（带有标记的点）
def plot_constellation_with_labels(constellation_points, title="标记星座图"):
    # 将复数星座点转换为实部和虚部
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    plt.figure(figsize=(10, 8))
    
    # 绘制星座点
    plt.scatter(real_parts, imag_parts, color='blue', marker='o', s=50)
    
    # 为每个点添加标签
    for i, (r, iq) in enumerate(zip(real_parts, imag_parts)):
        plt.annotate(f'{i}', (r, iq), xytext=(5, 5), textcoords='offset points')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # 保存图表
    plt.savefig('constellation_with_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("标记星座图已保存为 'constellation_with_labels.png'")

# 绘制星座点的星座能量分布
def plot_constellation_energy_distribution(constellation_points, title="星座能量分布"):
    # 计算每个星座点的能量
    energies = tf.abs(constellation_points)**2
    
    plt.figure(figsize=(10, 6))
    
    # 绘制能量分布直方图
    plt.hist(energies.numpy(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加统计线
    plt.axvline(tf.reduce_mean(energies), color='red', linestyle='dashed', linewidth=2, label=f'平均能量: {tf.reduce_mean(energies):.2f}')
    plt.axvline(tf.reduce_min(energies), color='green', linestyle='dashed', linewidth=2, label=f'最小能量: {tf.reduce_min(energies):.2f}')
    plt.axvline(tf.reduce_max(energies), color='purple', linestyle='dashed', linewidth=2, label=f'最大能量: {tf.reduce_max(energies):.2f}')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('能量')
    plt.ylabel('频数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    plt.savefig('constellation_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("星座能量分布图已保存为 'constellation_energy_distribution.png'")

# 绘制星座点的星座距离分布
def plot_constellation_distance_distribution(constellation_points, title="星座点距离分布"):
    # 将复数星座点转换为实部和虚部的数组
    real_parts = tf.math.real(constellation_points).numpy()
    imag_parts = tf.math.imag(constellation_points).numpy()
    
    # 计算所有点对之间的距离
    num_points = len(real_parts)
    distances = []
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            real_diff = real_parts[i] - real_parts[j]
            imag_diff = imag_parts[i] - imag_parts[j]
            distance = np.sqrt(real_diff**2 + imag_diff**2)
            distances.append(distance)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制距离分布直方图
    plt.hist(distances, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    
    # 添加统计线
    min_dist = np.min(distances)
    avg_dist = np.mean(distances)
    
    plt.axvline(min_dist, color='red', linestyle='dashed', linewidth=2, label=f'最小距离: {min_dist:.3f}')
    plt.axvline(avg_dist, color='blue', linestyle='dashed', linewidth=2, label=f'平均距离: {avg_dist:.3f}')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('点间距离')
    plt.ylabel('频数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    plt.savefig('constellation_distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("星座点距离分布图已保存为 'constellation_distance_distribution.png'")

# 比较两个星座图
def compare_constellations(constellation1, constellation2, name1="原始星座", name2="优化后星座"):
    plt.figure(figsize=(12, 6))
    
    # 绘制第一个星座
    plt.subplot(1, 2, 1)
    plt.scatter(tf.math.real(constellation1).numpy(), 
                tf.math.imag(constellation1).numpy(),
                color='blue', marker='o')
    plt.title(name1)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    # 绘制第二个星座
    plt.subplot(1, 2, 2)
    plt.scatter(tf.math.real(constellation2).numpy(), 
                tf.math.imag(constellation2).numpy(),
                color='red', marker='x')
    plt.title(name2)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("星座比较图已保存为 'constellation_comparison.png'")

# 生成星座分析报告
def generate_constellation_report(constellation_points, report_path="constellation_analysis_report.txt"):
    # 计算各种统计指标
    stats = calculate_constellation_stats(constellation_points)
    min_distance = calculate_min_distance(constellation_points)
    distribution_analysis = analyze_constellation_distribution(constellation_points)
    
    # 生成报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("星座点分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 基本统计特征\n")
        f.write(f"   - 实部均值: {stats['mean_real']:.6f}\n")
        f.write(f"   - 虚部均值: {stats['mean_imag']:.6f}\n")
        f.write(f"   - 实部标准差: {stats['std_real']:.6f}\n")
        f.write(f"   - 虚部标准差: {stats['std_imag']:.6f}\n")
        f.write(f"   - 最大幅度: {stats['max_amplitude']:.6f}\n")
        f.write(f"   - 最小幅度: {stats['min_amplitude']:.6f}\n")
        f.write(f"   - 平均幅度: {stats['avg_amplitude']:.6f}\n\n")
        
        f.write("2. 点间距离特征\n")
        f.write(f"   - 最小点间距离: {min_distance:.6f}\n\n")
        
        f.write("3. 分布特征\n")
        f.write(f"   - 是否规则网格: {distribution_analysis['is_regular_grid']}\n")
        f.write(f"   - 唯一实部数量: {distribution_analysis['num_unique_real']}\n")
        f.write(f"   - 唯一虚部数量: {distribution_analysis['num_unique_imag']}\n")
        f.write(f"   - 分布熵: {distribution_analysis['entropy']:.6f}\n")
        f.write(f"   - 最大熵: {distribution_analysis['max_entropy']:.6f}\n")
        f.write(f"   - 熵效率: {distribution_analysis['entropy_efficiency']:.6f}\n\n")
        
        f.write("4. 结论\n")
        if distribution_analysis['is_regular_grid']:
            f.write("   - 该星座点呈现规则网格分布，类似于标准QAM星座。\n")
        else:
            f.write("   - 该星座点呈现非规则分布，可能是经过优化的星座。\n")
        
        if min_distance > 0.1:
            f.write("   - 星座点之间保持了良好的最小距离，有利于噪声环境下的区分。\n")
        else:
            f.write("   - 星座点之间的最小距离较小，可能在噪声环境下容易混淆。\n")
        
        if distribution_analysis['entropy_efficiency'] > 0.8:
            f.write("   - 星座点分布较为均匀，熵效率较高。\n")
        else:
            f.write("   - 星座点分布不够均匀，熵效率较低。\n")
    
    print(f"星座分析报告已保存为 '{report_path}'")

# 主函数
def main():
    print("星座分析工具")
    print("=" * 50)
    
    # 设置参数
    NUM_BITS_PER_SYMBOL = 6  # 64-QAM
    WEIGHTS_PATH = "constellation_weights"
    
    # 创建原始QAM星座
    original_constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    original_points = original_constellation.points
    
    # 尝试加载预训练的星座点
    loaded_points = load_pretrained_constellation(WEIGHTS_PATH, NUM_BITS_PER_SYMBOL)
    
    if loaded_points is not None:
        # 创建自定义星座
        custom_constellation_layer = CustomConstellation(NUM_BITS_PER_SYMBOL, loaded_points)
        optimized_points = custom_constellation_layer()
        
        # 比较原始星座和优化后的星座
        print("\n比较原始星座和优化后的星座:")
        compare_constellations(original_points, optimized_points, "原始64-QAM星座", "优化后的星座")
        
        # 分析优化后的星座
        print("\n分析优化后的星座:")
        generate_constellation_report(optimized_points, "optimized_constellation_report.txt")
        
        # 绘制各种图表
        plot_constellation_heatmap(optimized_points, "优化后星座点分布热力图")
        plot_constellation_with_labels(optimized_points, "优化后标记星座图")
        plot_constellation_energy_distribution(optimized_points, "优化后星座能量分布")
        plot_constellation_distance_distribution(optimized_points, "优化后星座点距离分布")
    else:
        # 如果没有加载到预训练星座，则分析原始星座
        print("\n分析原始QAM星座:")
        generate_constellation_report(original_points, "original_constellation_report.txt")
        
        # 绘制各种图表
        plot_constellation_heatmap(original_points, "原始星座点分布热力图")
        plot_constellation_with_labels(original_points, "原始标记星座图")
        plot_constellation_energy_distribution(original_points, "原始星座能量分布")
        plot_constellation_distance_distribution(original_points, "原始星座点距离分布")
    
    print("\n星座分析完成！")

# 运行主函数
if __name__ == "__main__":
    main()