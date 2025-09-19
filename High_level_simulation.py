import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

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

# 高级QAM星座类，支持各种优化技术
class AdvancedQAMConstellation(tf.keras.layers.Layer):
    def __init__(self, num_bits_per_symbol, optimization_type=None):
        super(AdvancedQAMConstellation, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        self.optimization_type = optimization_type
        
        # 初始化星座点
        self._initialize_constellation()
    
    def _initialize_constellation(self):
        # 创建基本QAM星座
        qam_constellation = sn.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        initial_points = tf.stack([tf.math.real(qam_constellation.points),
                                   tf.math.imag(qam_constellation.points)], axis=0)
        
        # 根据优化类型调整初始星座
        if self.optimization_type == "power_allocation":
            # 功率分配优化：为边缘点分配更多功率
            idx = tf.range(self.num_points)
            # 计算每个点到原点的距离
            distances = tf.sqrt(initial_points[0]**2 + initial_points[1]** 2)
            # 基于距离的权重，距离越远权重越大
            weights = 1.0 + 0.5 * distances / tf.reduce_max(distances)
            # 应用权重
            weighted_points = tf.stack([initial_points[0] * weights, initial_points[1] * weights], axis=0)
            self.points = tf.Variable(weighted_points, dtype=tf.float32)
        elif self.optimization_type == "rotated":
            # 旋转优化：将星座旋转45度
            theta = np.pi / 4  # 45度
            rotation_matrix = tf.constant([[np.cos(theta), -np.sin(theta)],
                                           [np.sin(theta), np.cos(theta)]], dtype=tf.float32)
            # 重塑星座点以应用旋转矩阵
            points_reshaped = tf.transpose(initial_points)
            rotated_points = tf.linalg.matvec(rotation_matrix, points_reshaped)
            rotated_points = tf.transpose(rotated_points)
            self.points = tf.Variable(rotated_points, dtype=tf.float32)
        elif self.optimization_type == "staggered":
            # 交错优化：将偶数行略微偏移
            points = initial_points.numpy()
            # 找到所有行
            unique_rows = np.unique(np.round(points[1], decimals=3))
            # 为偶数行添加偏移
            for i, row in enumerate(unique_rows):
                if i % 2 == 1:  # 偶数行（索引从0开始）
                    row_mask = np.isclose(points[1], row, atol=1e-3)
                    points[0, row_mask] += 0.2
            self.points = tf.Variable(points, dtype=tf.float32)
        else:
            # 标准QAM星座
            self.points = tf.Variable(initial_points, dtype=tf.float32)
    
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

# 通信系统模拟类
class CommunicationSystem:
    def __init__(self, num_bits_per_symbol, constellation_type="qam", 
                 coding_rate=1.0, channel_model="awgn"):
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coding_rate = coding_rate
        self.channel_model = channel_model
        
        # 创建二进制源
        self.binary_source = sn.phy.mapping.BinarySource()
        
        # 创建星座和映射器
        if constellation_type == "qam":
            self.constellation = sn.phy.mapping.Constellation("qam", num_bits_per_symbol)
        elif constellation_type == "psk":
            self.constellation = sn.phy.mapping.Constellation("psk", num_bits_per_symbol)
        elif constellation_type.startswith("advanced_"):
            # 高级优化星座
            opt_type = constellation_type.replace("advanced_", "")
            self.constellation_layer = AdvancedQAMConstellation(num_bits_per_symbol, opt_type)
            constellation_points = self.constellation_layer()
            self.constellation = sn.phy.mapping.Constellation(
                "custom",
                num_bits_per_symbol=num_bits_per_symbol,
                points=constellation_points,
                normalize=False,
                center=False
            )
        else:
            raise ValueError(f"不支持的星座类型: {constellation_type}")
        
        # 创建映射器和解映射器
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.phy.mapping.Demapper("app", constellation=self.constellation)
        
        # 创建信道模型
        if channel_model == "awgn":
            self.channel = sn.phy.channel.AWGN()
        elif channel_model == "rayleigh":
            self.channel = sn.phy.channel.RayleighBlockFading(
                num_rx=1, num_rx_ant=1,
                fading_freq=0.01, # 多普勒频移
                block_length=1000 # 块长度
            )
        else:
            raise ValueError(f"不支持的信道模型: {channel_model}")
        
        # 创建解码器（如果编码率小于1）
        self.decoder = None
        self.encoder = None
        if coding_rate < 1.0:
            # 使用Turbo码作为示例
            self.encoder = sn.fec.turbo.TurboEncoder(
                constraint_length=4,
                generator_poly=[[13, 15], [17, 15]],
                puncture_pattern=[[1, 1, 0], [1, 0, 1]]
            )
            self.decoder = sn.fec.turbo.TurboDecoder(
                self.encoder, 
                num_iter=6  # 解码迭代次数
            )
    
    def simulate(self, batch_size, ebno_db):
        # 计算噪声功率
        no = sn.phy.utils.ebnodb2no(ebno_db,
                                   num_bits_per_symbol=self.num_bits_per_symbol,
                                   coderate=self.coding_rate)
        
        # 生成随机比特
        if self.encoder is not None:
            # 如果有编码器，生成信息比特
            info_bits = self.binary_source([batch_size, self.encoder.k])
            # 编码
            bits = self.encoder(info_bits)
        else:
            # 无编码传输
            bits = self.binary_source([batch_size, 1000])
        
        # 映射为星座点
        x = self.mapper(bits)
        
        # 通过信道传输
        y = self.channel(x, no)
        
        # 解调得到LLR
        llr = self.demapper(y, no)
        
        # 如果有解码器，进行解码
        if self.decoder is not None:
            llr = self.decoder(llr)
            # 计算BER时只比较信息比特
            bits_to_compare = info_bits
        else:
            bits_to_compare = bits
        
        # 计算BER
        ber = sn.phy.utils.ber(bits_to_compare, llr > 0)
        
        # 计算SER（符号错误率）
        # 首先将比特转换为符号索引
        num_symbols = tf.shape(bits)[1] // self.num_bits_per_symbol
        bits_reshaped = tf.reshape(bits, [batch_size, num_symbols, self.num_bits_per_symbol])
        symbol_indices = sn.utils.binary_source.bits_to_indices(bits_reshaped)
        
        # 将LLR转换为硬判决符号索引
        llr_reshaped = tf.reshape(llr, [batch_size, num_symbols, self.num_bits_per_symbol])
        hard_decisions = tf.cast(llr_reshaped > 0, tf.int32)
        detected_indices = sn.utils.binary_source.bits_to_indices(hard_decisions)
        
        # 计算SER
        ser = tf.reduce_mean(tf.cast(symbol_indices != detected_indices, tf.float32))
        
        return ber, ser

# 高级通信系统模拟器
class AdvancedSystemSimulator:
    def __init__(self):
        self.systems = {}
    
    def add_system(self, name, system):
        """添加一个通信系统到模拟器"""
        self.systems[name] = system
    
    def run_simulation(self, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=200):
        """运行系统性能模拟"""
        results = {}
        
        # 为每个系统运行模拟
        for name, system in self.systems.items():
            print(f"\n开始模拟系统: {name}")
            bers = []
            sers = []
            
            for ebno_db in ebno_dbs:
                print(f"  模拟Eb/No = {ebno_db} dB")
                
                # 用于统计的变量
                total_errors = 0
                total_bits = 0
                total_symbol_errors = 0
                total_symbols = 0
                mc_iter = 0
                
                # 使用tqdm显示进度
                pbar = tqdm(total=num_target_block_errors, desc=f"    累积错误数")
                
                while total_errors < num_target_block_errors and mc_iter < max_mc_iter:
                    # 运行一次模拟
                    ber, ser = system.simulate(batch_size, ebno_db)
                    
                    # 计算本次迭代的错误数
                    current_bits = batch_size * 1000  # 假设每次模拟1000比特
                    current_errors = ber * current_bits
                    current_symbol_errors = ser * (current_bits // system.num_bits_per_symbol)
                    
                    # 更新统计
                    total_errors += current_errors
                    total_bits += current_bits
                    total_symbol_errors += current_symbol_errors
                    total_symbols += current_bits // system.num_bits_per_symbol
                    mc_iter += 1
                    
                    # 更新进度条
                    pbar.update(min(current_errors, num_target_block_errors - pbar.n))
                
                pbar.close()
                
                # 计算最终的BER和SER
                final_ber = total_errors / total_bits
                final_ser = total_symbol_errors / total_symbols
                
                bers.append(final_ber)
                sers.append(final_ser)
                
                print(f"    完成 - BER: {final_ber:.6f}, SER: {final_ser:.6f}")
            
            results[name] = {"ber": np.array(bers), "ser": np.array(sers)}
        
        return results
    
    def plot_results(self, results, ebno_dbs, plot_type="ber"):
        """绘制模拟结果"""
        plt.figure(figsize=(10, 6))
        
        for name, data in results.items():
            if plot_type.lower() == "ber":
                plt.semilogy(ebno_dbs, data["ber"], marker='o', label=name)
                plt.ylabel('误比特率 (BER)')
            else:
                plt.semilogy(ebno_dbs, data["ser"], marker='s', label=name)
                plt.ylabel('误符号率 (SER)')
            
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Eb/No (dB)')
        plt.title(f'不同通信系统的{"BER" if plot_type.lower() == "ber" else "SER"}性能比较')
        plt.legend()
        
        # 保存图表
        filename = f'system_performance_{plot_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能比较图已保存为 '{filename}'")

# 运行高级通信系统模拟
def run_high_level_simulation():
    print("高级通信系统性能模拟")
    print("=" * 50)
    
    # 创建模拟器
    simulator = AdvancedSystemSimulator()
    
    # 设置参数
    NUM_BITS_PER_SYMBOL = 6  # 64-QAM
    CODING_RATE = 1.0  # 无编码传输
    
    # 添加不同的通信系统
    print("添加不同的通信系统...")
    
    # 标准QAM系统
    standard_qam = CommunicationSystem(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        constellation_type="qam",
        coding_rate=CODING_RATE,
        channel_model="awgn"
    )
    simulator.add_system("标准64-QAM", standard_qam)
    
    # 功率分配优化的QAM系统
    power_alloc_qam = CommunicationSystem(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        constellation_type="advanced_power_allocation",
        coding_rate=CODING_RATE,
        channel_model="awgn"
    )
    simulator.add_system("功率分配优化64-QAM", power_alloc_qam)
    
    # 旋转优化的QAM系统
    rotated_qam = CommunicationSystem(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        constellation_type="advanced_rotated",
        coding_rate=CODING_RATE,
        channel_model="awgn"
    )
    simulator.add_system("旋转优化64-QAM", rotated_qam)
    
    # 交错优化的QAM系统
    staggered_qam = CommunicationSystem(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        constellation_type="advanced_staggered",
        coding_rate=CODING_RATE,
        channel_model="awgn"
    )
    simulator.add_system("交错优化64-QAM", staggered_qam)
    
    # 创建Eb/No值数组
    ebno_dbs = np.linspace(12.0, 21.0, 10)
    
    # 运行模拟
    print("\n开始运行系统性能模拟...")
    results = simulator.run_simulation(
        ebno_dbs=ebno_dbs,
        batch_size=256,
        num_target_block_errors=100,
        max_mc_iter=200
    )
    
    # 绘制BER性能结果
    print("\n绘制BER性能比较图...")
    simulator.plot_results(results, ebno_dbs, plot_type="ber")
    
    # 绘制SER性能结果
    print("\n绘制SER性能比较图...")
    simulator.plot_results(results, ebno_dbs, plot_type="ser")
    
    print("\n高级通信系统性能模拟完成！")
    print("生成的文件：")
    print("- system_performance_ber.png: BER性能比较图")
    print("- system_performance_ser.png: SER性能比较图")

# 主函数，提供命令行接口
def main():
    print("高级通信系统模拟工具")
    print("=" * 50)
    
    # 简单的命令行参数解析
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("使用方法:")
            print("  python High_level_simulation.py [选项]")
            print("\n选项:")
            print("  --help, -h            显示帮助信息")
            print("  --quick               快速模拟（较少的Eb/No点和迭代次数）")
            print("  --extensive           扩展模拟（更多的Eb/No点和迭代次数）")
            print("  --rayleigh            使用Rayleigh衰落信道（默认是AWGN信道）")
            sys.exit(0)
    
    # 运行高级系统模拟
    run_high_level_simulation()

# 运行主函数
if __name__ == "__main__":
    main()