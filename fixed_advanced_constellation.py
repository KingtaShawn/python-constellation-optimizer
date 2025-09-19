import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense

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

# 定义星座点优化的参数
NUM_BITS_PER_SYMBOL = 6  # 64-QAM
BATCH_SIZE = 256  # 增大批次大小以提高训练稳定性
EBN0_DB_MIN = 12.0  # 起始信噪比
EBN0_DB_MAX = 21.0  # 终止信噪比
NUM_EBN0_POINTS = 10  # 信噪比点数
NUM_TRAINING_ITERATIONS = 30000  # 增加训练迭代次数
INITIAL_LEARNING_RATE = 0.0005  # 减小初始学习率
DECAY_STEPS = 10000  # 学习率衰减步数
DECAY_RATE = 0.5  # 学习率衰减率

# 自定义星座类，支持松弛约束
class CustomConstellation(Layer):
    def __init__(self, num_bits_per_symbol):
        super(CustomConstellation, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        
        # 初始化星座点，基于QAM星座但增加扰动空间
        qam_constellation = sn.phy.mapping.Constellation("qam", num_bits_per_symbol)
        initial_points = tf.stack([tf.math.real(qam_constellation.points),
                                   tf.math.imag(qam_constellation.points)], axis=0)
        
        # 使用tf.Variable创建可训练的星座点
        self.points = tf.Variable(initial_points, dtype=tf.float32)
        
        # 松弛约束参数，用于控制星座点的分散程度
        self.relaxation_factor = tf.Variable(0.1, dtype=tf.float32, trainable=True)
    
    def call(self, inputs=None):
        # 归一化星座点，保持能量恒定
        normalized_points = self.normalize_constellation(self.points)
        
        # 将星座点转换为复数形式
        complex_points = tf.complex(normalized_points[0], normalized_points[1])
        
        # 返回星座点和松弛约束因子
        return complex_points, self.relaxation_factor
    
    def normalize_constellation(self, points):
        # 计算星座点的平均能量
        power = tf.reduce_mean(tf.square(points[0]) + tf.square(points[1]))
        
        # 归一化星座点，使其平均能量为1
        normalized_points = points / tf.sqrt(power)
        
        return normalized_points

# 带距离正则化的损失函数
def custom_loss_function(bits, llr, constellation_points, relaxation_factor):
    # 基础的二元交叉熵损失
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    base_loss = tf.reduce_mean(bce(bits, llr))
    
    # 计算星座点之间的最小距离
    min_distance = calculate_min_distance(constellation_points)
    
    # 距离正则化项，鼓励星座点之间保持足够的距离
    distance_regularization = 0.1 * (1.0 - min_distance) * (1.0 + relaxation_factor)
    
    # 综合损失
    total_loss = base_loss + distance_regularization
    
    return total_loss, base_loss, distance_regularization

# 计算星座点之间的最小距离
def calculate_min_distance(constellation_points):
    # 将复数星座点转换为实部和虚部的数组
    real_parts = tf.math.real(constellation_points)
    imag_parts = tf.math.imag(constellation_points)
    
    # 创建所有点对的索引
    indices = tf.range(tf.shape(constellation_points)[0])
    idx1, idx2 = tf.meshgrid(indices, indices, indexing='ij')
    
    # 只考虑i < j的点对，避免重复计算和自己与自己的距离
    mask = idx1 < idx2
    idx1 = tf.boolean_mask(idx1, mask)
    idx2 = tf.boolean_mask(idx2, mask)
    
    # 计算所有点对之间的欧几里得距离
    real_diff = tf.gather(real_parts, idx1) - tf.gather(real_parts, idx2)
    imag_diff = tf.gather(imag_parts, idx1) - tf.gather(imag_parts, idx2)
    distances = tf.sqrt(tf.square(real_diff) + tf.square(imag_diff))
    
    # 找到最小距离
    min_distance = tf.reduce_min(distances)
    
    return min_distance

# 课程学习调度器
class CurriculumScheduler:
    def __init__(self, initial_ebno, final_ebno, steps, batch_size):
        self.initial_ebno = initial_ebno
        self.final_ebno = final_ebno
        self.steps = steps
        self.batch_size = batch_size
        self.current_step = 0
    
    def get_ebno(self):
        # 线性增加SNR值
        progress = min(self.current_step / self.steps, 1.0)
        current_ebno = self.initial_ebno + progress * (self.final_ebno - self.initial_ebno)
        
        # 为每个批次样本生成略微不同的SNR值，增加训练的鲁棒性
        ebno_per_sample = current_ebno + tf.random.uniform([self.batch_size], minval=-0.5, maxval=0.5)
        
        self.current_step += 1
        
        return ebno_per_sample

# 端到端可微分通信系统模型
class End2EndSystem(Model):
    def __init__(self, num_bits_per_symbol, training=True):
        super(End2EndSystem, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.training = training
        
        # 自定义可训练星座
        self.custom_constellation = CustomConstellation(num_bits_per_symbol)
        
        # Sionna通信系统组件
        self.binary_source = sn.phy.mapping.BinarySource()
        self.mapper = None  # 稍后初始化
        self.demapper = None  # 稍后初始化
        self.awgn_channel = sn.phy.channel.AWGN()
        
        # 初始化映射器和解调器
        self._initialize_mapper_demapper()
    
    def _initialize_mapper_demapper(self):
        # 获取星座点
        constellation_points, _ = self.custom_constellation()
        
        # 创建Sionna星座对象
        sionna_constellation = sn.phy.mapping.Constellation("custom",
                                                           num_bits_per_symbol=self.num_bits_per_symbol,
                                                           points=constellation_points,
                                                           normalize=False,  # 自定义星座已经归一化
                                                           center=False)     # 自定义星座已经居中
        
        # 初始化映射器和解调器
        self.mapper = sn.phy.mapping.Mapper(constellation=sionna_constellation)
        self.demapper = sn.phy.mapping.Demapper("app", constellation=sionna_constellation)
    
    @tf.function(jit_compile=True)  # 启用XLA编译以提高性能
    def call(self, batch_size, ebno_db):
        # 获取当前训练的星座点
        constellation_points, relaxation_factor = self.custom_constellation()
        
        # 更新映射器和解调器的星座点
        self.mapper.constellation.points = constellation_points
        self.demapper.constellation.points = constellation_points
        
        # 计算噪声功率
        no = sn.phy.utils.ebnodb2no(ebno_db,
                                 num_bits_per_symbol=self.num_bits_per_symbol,
                                 coderate=1.0)  # 无编码传输
        
        # 生成随机比特
        bits = self.binary_source([batch_size, 1200])  # 块长度为1200比特
        
        # 映射为星座点
        x = self.mapper(bits)
        
        # 通过AWGN信道传输
        y = self.awgn_channel(x, no)
        
        # 解调得到LLR
        llr = self.demapper(y, no)
        
        if self.training:
            # 计算带正则化的损失
            total_loss, base_loss, distance_reg = custom_loss_function(
                bits, llr, constellation_points, relaxation_factor
            )
            return total_loss, base_loss, distance_reg
        else:
            # 推理模式下返回比特和LLR
            return bits, llr

# 训练高级星座模型
def train_advanced_constellation():
    print("开始训练高级星座优化模型...")
    
    # 创建端到端系统模型
    model = End2EndSystem(NUM_BITS_PER_SYMBOL, training=True)
    
    # 创建指数衰减学习率调度器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    
    # 使用Adam优化器，带有学习率调度
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # 创建课程学习调度器
    curriculum = CurriculumScheduler(
        initial_ebno=EBN0_DB_MIN, 
        final_ebno=EBN0_DB_MAX, 
        steps=NUM_TRAINING_ITERATIONS, 
        batch_size=BATCH_SIZE
    )
    
    # 创建损失记录数组
    total_losses = np.zeros(NUM_TRAINING_ITERATIONS)
    base_losses = np.zeros(NUM_TRAINING_ITERATIONS)
    distance_regs = np.zeros(NUM_TRAINING_ITERATIONS)
    learning_rates = np.zeros(NUM_TRAINING_ITERATIONS)
    snr_values = np.zeros(NUM_TRAINING_ITERATIONS)
    
    # 训练循环
    for i in range(NUM_TRAINING_ITERATIONS):
        # 获取当前学习率
        current_lr = lr_schedule(optimizer.iterations).numpy()
        
        # 获取当前SNR值（通过课程学习调度器）
        ebno_db = curriculum.get_ebno()
        
        # 记录当前SNR值（取平均值）
        current_snr = tf.reduce_mean(ebno_db).numpy()
        
        with tf.GradientTape() as tape:
            # 前向传播
            total_loss, base_loss, distance_reg = model(BATCH_SIZE, ebno_db)
        
        # 计算梯度
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # 应用梯度裁剪，防止梯度爆炸
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # 更新模型参数
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        
        # 记录损失值
        total_losses[i] = total_loss.numpy()
        base_losses[i] = base_loss.numpy()
        distance_regs[i] = distance_reg.numpy()
        learning_rates[i] = current_lr
        snr_values[i] = current_snr
        
        # 每100次迭代打印一次进度
        if i % 100 == 0:
            print(f"迭代 {i}/{NUM_TRAINING_ITERATIONS} - 总损失: {total_loss:.4f}, \
                  基础损失: {base_loss:.4f}, 距离正则化: {distance_reg:.4f}, \
                  学习率: {current_lr:.6f}, SNR: {current_snr:.2f} dB")
    
    # 保存训练好的模型权重
    model.save_weights("advanced_constellation_weights")
    print("模型权重已保存为 'advanced_constellation_weights'")
    
    # 绘制训练过程中的损失曲线
    plot_training_progress(total_losses, base_losses, distance_regs, 
                          learning_rates, snr_values)
    
    # 绘制优化后的星座图
    plot_constellation_comparison(model)
    
    return model

# 绘制训练进度图表
def plot_training_progress(total_losses, base_losses, distance_regs, 
                          learning_rates, snr_values):
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(total_losses, label='总损失')
    plt.plot(base_losses, label='基础损失')
    plt.plot(distance_regs, label='距离正则化项')
    plt.yscale('log')
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    
    # 绘制学习率曲线
    plt.subplot(3, 1, 2)
    plt.plot(learning_rates)
    plt.yscale('log')
    plt.title('学习率变化')
    plt.xlabel('迭代次数')
    plt.ylabel('学习率')
    plt.grid(True)
    
    # 绘制SNR变化曲线
    plt.subplot(3, 1, 3)
    plt.plot(snr_values)
    plt.title('SNR值变化（课程学习）')
    plt.xlabel('迭代次数')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练进度图表已保存为 'training_progress.png'")

# 绘制优化后的星座图并与原始星座图比较
def plot_constellation_comparison(model):
    # 获取优化后的星座点
    optimized_points, _ = model.custom_constellation()
    
    # 创建原始QAM星座进行比较
    original_constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制原始QAM星座
    plt.subplot(1, 2, 1)
    plt.scatter(tf.math.real(original_constellation.points), 
                tf.math.imag(original_constellation.points),
                color='blue', marker='o')
    plt.title('原始64-QAM星座')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    # 绘制优化后的星座
    plt.subplot(1, 2, 2)
    plt.scatter(tf.math.real(optimized_points), 
                tf.math.imag(optimized_points),
                color='red', marker='x')
    plt.title('优化后的星座')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("星座比较图已保存为 'constellation_comparison.png'")

# 评估模型性能（计算BER）
def evaluate_model_performance(model, original_constellation, ebno_dbs):
    print("评估模型性能...")
    
    # 创建用于评估的模型实例
    eval_model = End2EndSystem(NUM_BITS_PER_SYMBOL, training=False)
    eval_model.set_weights(model.get_weights())
    
    # 创建BER评估工具
    ber_evaluator = sn.phy.utils.PlotBER("星座优化性能比较")
    
    # 定义评估函数
    def evaluate_optimized(batch_size, ebno_db):
        bits, llr = eval_model(batch_size, ebno_db)
        return bits, llr
    
    def evaluate_original(batch_size, ebno_db):
        # 创建原始QAM系统
        binary_source = sn.phy.mapping.BinarySource()
        mapper = sn.phy.mapping.Mapper(constellation=original_constellation)
        demapper = sn.phy.mapping.Demapper("app", constellation=original_constellation)
        awgn_channel = sn.phy.channel.AWGN()
        
        # 计算噪声功率
        no = sn.phy.utils.ebnodb2no(ebno_db,
                                 num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                 coderate=1.0)
        
        # 生成比特并通过系统
        bits = binary_source([batch_size, 1200])
        x = mapper(bits)
        y = awgn_channel(x, no)
        llr = demapper(y, no)
        
        return bits, llr
    
    # 评估优化后的星座性能
    ber_evaluator.simulate(
        evaluate_optimized,
        ebno_dbs=ebno_dbs,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        legend="优化后的星座",
        soft_estimates=True,
        max_mc_iter=200,
        show_fig=False
    )
    
    # 评估原始QAM星座性能
    ber_evaluator.simulate(
        evaluate_original,
        ebno_dbs=ebno_dbs,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        legend="原始64-QAM",
        soft_estimates=True,
        max_mc_iter=200,
        show_fig=True
    )
    
    # 保存BER性能比较图
    plt.savefig('ber_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("BER性能比较图已保存为 'ber_comparison.png'")

# 主函数
if __name__ == "__main__":
    print("高级星座优化与性能分析工具")
    print("=" * 50)
    
    # 训练高级星座模型
    model = train_advanced_constellation()
    
    # 创建原始QAM星座用于比较
    original_constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    
    # 生成评估用的SNR值数组
    ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, NUM_EBN0_POINTS)
    
    # 评估模型性能
    evaluate_model_performance(model, original_constellation, ebno_dbs)
    
    print("星座优化与性能分析完成！")
    print("生成的文件：")
    print("- advanced_constellation_weights: 训练好的模型权重")
    print("- training_progress.png: 训练进度图表")
    print("- constellation_comparison.png: 星座比较图")
    print("- ber_comparison.png: BER性能比较图")