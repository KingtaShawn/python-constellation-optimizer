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

# 高级星座类，支持可训练星座点和多种优化方法
class AdvancedConstellation(tf.keras.layers.Layer):
    def __init__(self, num_bits_per_symbol, constellation_type='qam', 
                 trainable=True, regularize_distance=True, distance_weight=0.1):
        super(AdvancedConstellation, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        self.constellation_type = constellation_type
        self.trainable = trainable
        self.regularize_distance = regularize_distance
        self.distance_weight = distance_weight
        
        # 初始化星座点
        self._initialize_constellation()
    
    def _initialize_constellation(self):
        # 根据星座类型初始化星座点
        if self.constellation_type.lower() == 'qam':
            # 创建标准QAM星座作为初始值
            qam_constellation = sn.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
            self.points = tf.Variable(
                tf.stack([tf.math.real(qam_constellation.points), 
                         tf.math.imag(qam_constellation.points)], axis=0),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.constellation_type.lower() == 'psk':
            # 创建PSK星座作为初始值
            psk_constellation = sn.phy.mapping.Constellation("psk", self.num_bits_per_symbol)
            self.points = tf.Variable(
                tf.stack([tf.math.real(psk_constellation.points), 
                         tf.math.imag(psk_constellation.points)], axis=0),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.constellation_type.lower() == 'custom':
            # 创建自定义星座，使用随机初始化
            # 确保实部和虚部分布在[-sqrt(M-1), sqrt(M-1)]范围内
            scale_factor = tf.sqrt(tf.cast(self.num_points - 1, tf.float32))
            # 使用均匀分布初始化
            real_part = tf.random.uniform([self.num_points], -scale_factor, scale_factor)
            imag_part = tf.random.uniform([self.num_points], -scale_factor, scale_factor)
            self.points = tf.Variable(
                tf.stack([real_part, imag_part], axis=0),
                trainable=self.trainable,
                dtype=tf.float32
            )
        else:
            raise ValueError(f"不支持的星座类型: {self.constellation_type}")
    
    def call(self, inputs=None):
        # 归一化星座点，使其平均能量为1
        normalized_points = self.normalize_constellation(self.points)
        
        # 计算距离正则项（如果启用）
        if self.regularize_distance and self.trainable:
            distance_loss = self._calculate_distance_loss(normalized_points)
            self.add_loss(self.distance_weight * distance_loss)
        
        # 将星座点转换为复数形式
        complex_points = tf.complex(normalized_points[0], normalized_points[1])
        
        return complex_points
    
    def normalize_constellation(self, points):
        # 计算星座点的平均能量
        power = tf.reduce_mean(tf.square(points[0]) + tf.square(points[1]))
        
        # 归一化星座点，使其平均能量为1
        normalized_points = points / tf.sqrt(power)
        
        return normalized_points
    
    def _calculate_distance_loss(self, points):
        # 计算所有星座点对之间的最小距离
        # 获取实部和虚部
        real_part = points[0]
        imag_part = points[1]
        
        # 计算所有点对之间的距离
        # 为了高效计算，使用广播机制
        real_diff = real_part[:, tf.newaxis] - real_part[tf.newaxis, :]
        imag_diff = imag_part[:, tf.newaxis] - imag_part[tf.newaxis, :]
        distances = tf.sqrt(tf.square(real_diff) + tf.square(imag_diff))
        
        # 排除对角线元素（点到自身的距离）
        mask = 1.0 - tf.eye(tf.shape(distances)[0])
        masked_distances = distances * mask
        
        # 找到最小距离
        min_distance = tf.reduce_min(tf.boolean_mask(masked_distances, masked_distances > 0))
        
        # 距离损失：鼓励最小距离尽可能大
        # 使用负的最小距离作为损失，这样优化器会最大化最小距离
        distance_loss = -min_distance
        
        return distance_loss
    
    def get_min_distance(self):
        # 获取当前星座点的最小距离
        normalized_points = self.normalize_constellation(self.points)
        real_part = normalized_points[0]
        imag_part = normalized_points[1]
        
        real_diff = real_part[:, tf.newaxis] - real_part[tf.newaxis, :]
        imag_diff = imag_part[:, tf.newaxis] - imag_part[tf.newaxis, :]
        distances = tf.sqrt(tf.square(real_diff) + tf.square(imag_diff))
        
        mask = 1.0 - tf.eye(tf.shape(distances)[0])
        masked_distances = distances * mask
        
        min_distance = tf.reduce_min(tf.boolean_mask(masked_distances, masked_distances > 0))
        
        return min_distance.numpy()
    
    def visualize(self, title=None, save_path=None):
        # 可视化星座图
        complex_points = self.call()
        real_part = tf.math.real(complex_points).numpy()
        imag_part = tf.math.imag(complex_points).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(real_part, imag_part, s=100, c='blue', alpha=0.7)
        
        # 添加网格线和坐标轴
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 设置坐标轴范围
        max_val = max(max(abs(real_part)), max(abs(imag_part)))
        plt.xlim(-1.1*max_val, 1.1*max_val)
        plt.ylim(-1.1*max_val, 1.1*max_val)
        
        # 设置标题和标签
        if title:
            plt.title(title)
        else:
            plt.title(f'{self.constellation_type.upper()} 星座图 ({self.num_points}点)')
        
        plt.xlabel('实部')
        plt.ylabel('虚部')
        
        # 添加最小距离信息
        min_dist = self.get_min_distance()
        plt.text(0.05, 0.95, f'最小距离: {min_dist:.4f}', 
                 transform=plt.gca().transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # 保存图像（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"星座图已保存为: {save_path}")
        
        plt.tight_layout()
        plt.show()

# 端到端通信系统模型
class EndToEndSystem(tf.keras.Model):
    def __init__(self, num_bits_per_symbol, constellation_type='qam', trainable_constellation=True):
        super(EndToEndSystem, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        
        # 创建二进制源
        self.binary_source = sn.phy.mapping.BinarySource()
        
        # 创建高级星座
        self.constellation_layer = AdvancedConstellation(
            num_bits_per_symbol=num_bits_per_symbol,
            constellation_type=constellation_type,
            trainable=trainable_constellation,
            regularize_distance=True,
            distance_weight=0.05
        )
        
        # 创建映射器和解映射器
        # 注意：这里需要动态获取星座点，所以在build方法中初始化映射器和解映射器
        self.mapper = None
        self.demapper = None
        
        # 创建AWGN信道
        self.channel = sn.phy.channel.AWGN()
        
        # 创建损失函数
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def build(self, input_shape=None):
        # 获取星座点
        constellation_points = self.constellation_layer()
        
        # 创建星座对象
        self.constellation = sn.phy.mapping.Constellation(
            "custom",
            num_bits_per_symbol=self.num_bits_per_symbol,
            points=constellation_points,
            normalize=False,
            center=False
        )
        
        # 初始化映射器和解映射器
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.phy.mapping.Demapper("app", constellation=self.constellation)
    
    def call(self, inputs=None, ebno_db=10.0):
        # 如果没有输入，生成随机比特
        if inputs is None:
            batch_size = 1000  # 默认批量大小
            # 计算需要生成的比特数
            # 确保比特数是num_bits_per_symbol的整数倍
            num_symbols = 100  # 每个样本的符号数
            total_bits = batch_size * num_symbols * self.num_bits_per_symbol
            bits = self.binary_source([total_bits])
            bits = tf.reshape(bits, [batch_size, num_symbols * self.num_bits_per_symbol])
        else:
            bits = inputs
            batch_size = tf.shape(bits)[0]
        
        # 计算噪声功率
        no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol)
        
        # 获取星座点（这会触发距离正则化计算）
        self.constellation.points = self.constellation_layer()
        
        # 映射为星座点
        x = self.mapper(bits)
        
        # 通过信道传输
        y = self.channel(x, no)
        
        # 解调得到LLR
        llr = self.demapper(y, no)
        
        # 计算BER
        ber = sn.phy.utils.ber(bits, llr > 0)
        
        # 计算损失
        loss = self.bce_loss(bits, llr)
        
        # 添加额外的损失（来自星座点的距离正则化）
        for l in self.constellation_layer.losses:
            loss += l
        
        # 存储损失和BER以便训练过程中监控
        self.add_metric(ber, name='ber', aggregation='mean')
        
        return llr, loss
    
    def train_step(self, data):
        # 解包数据
        # 在这个简单实现中，我们忽略输入数据，使用内部生成的随机比特
        # 实际应用中可以根据需要修改
        
        # 获取当前的Eb/No值（可能是变化的）
        # 这里我们使用一个简单的课程学习策略
        current_epoch = self._get_current_epoch()
        # 从低Eb/No开始，逐渐增加到高Eb/No
        min_ebno = 5.0
        max_ebno = 20.0
        # 在前50%的训练中线性增加
        if current_epoch < self.epochs * 0.5:
            ebno_db = min_ebno + (max_ebno - min_ebno) * (current_epoch / (self.epochs * 0.5))
        else:
            ebno_db = max_ebno
        
        # 记录当前的Eb/No值
        tf.summary.scalar('ebno_db', ebno_db, step=self._train_counter)
        
        with tf.GradientTape() as tape:
            # 前向传播
            llr, loss = self.call(ebno_db=ebno_db)
            
            # 添加正则化损失
            loss += sum(self.losses)
        
        # 计算梯度
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 更新指标
        self.compiled_metrics.update_state(None, llr)
        
        # 返回结果
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}
    
    def _get_current_epoch(self):
        # 获取当前训练的轮次
        # 注意：这是一个简化实现，实际应用中可能需要调整
        if hasattr(self, '_train_counter'):
            return tf.cast(self._train_counter // self.steps_per_epoch, tf.float32)
        else:
            return 0.0
    
    def set_training_params(self, epochs, steps_per_epoch):
        # 设置训练参数
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # 初始化训练计数器
        self._train_counter = tf.Variable(0, dtype=tf.int64)
    
    def test_ber(self, ebno_dbs, batch_size=1000, num_target_block_errors=100):
        """测试不同Eb/No值下的BER性能"""
        bers = []
        
        for ebno_db in ebno_dbs:
            print(f"测试 Eb/No = {ebno_db} dB")
            
            # 用于统计的变量
            total_errors = 0
            total_bits = 0
            block_count = 0
            
            # 使用tqdm显示进度
            pbar = tqdm(total=num_target_block_errors, desc=" 累积错误数")
            
            while total_errors < num_target_block_errors and block_count < 1000:  # 防止无限循环
                # 生成随机比特
                num_symbols = 100  # 每个块的符号数
                bits = self.binary_source([batch_size, num_symbols * self.num_bits_per_symbol])
                
                # 计算噪声功率
                no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol)
                
                # 获取星座点
                self.constellation.points = self.constellation_layer()
                
                # 映射为星座点
                x = self.mapper(bits)
                
                # 通过信道传输
                y = self.channel(x, no)
                
                # 解调得到LLR
                llr = self.demapper(y, no)
                
                # 计算BER
                ber = sn.phy.utils.ber(bits, llr > 0)
                
                # 计算本次迭代的错误数
                current_bits = batch_size * num_symbols * self.num_bits_per_symbol
                current_errors = ber * current_bits
                
                # 更新统计
                total_errors += current_errors
                total_bits += current_bits
                block_count += 1
                
                # 更新进度条
                pbar.update(min(current_errors, num_target_block_errors - pbar.n))
            
            pbar.close()
            
            # 计算最终的BER
            final_ber = total_errors / total_bits
            bers.append(final_ber)
            
            print(f"  BER: {final_ber:.6f}")
        
        return np.array(bers)
    
    def plot_ber_performance(self, ebno_dbs, bers, baseline_bers=None, save_path=None):
        """绘制BER性能曲线"""
        plt.figure(figsize=(10, 6))
        
        # 绘制训练后的BER曲线
        plt.semilogy(ebno_dbs, bers, marker='o', linestyle='-', color='blue', label='优化后星座')
        
        # 如果提供了基准BER曲线，也绘制出来
        if baseline_bers is not None:
            plt.semilogy(ebno_dbs, baseline_bers, marker='s', linestyle='--', color='red', label='标准QAM')
        
        # 设置图表属性
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xlabel('Eb/No (dB)')
        plt.ylabel('误比特率 (BER)')
        plt.title(f'{self.num_points}-点星座图BER性能')
        plt.legend()
        
        # 保存图像（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BER性能图已保存为: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_constellation(self, title=None, save_path=None):
        """可视化星座图"""
        self.constellation_layer.visualize(title=title, save_path=save_path)

# 对比不同星座类型的函数
def compare_constellations(num_bits_per_symbol=6):
    """对比不同类型的星座图性能"""
    # 创建不同类型的星座
    constellation_types = ['qam', 'psk', 'custom']
    constellations = {}
    
    print(f"创建不同类型的{2**num_bits_per_symbol}-点星座图...")
    
    for const_type in constellation_types:
        constellation = AdvancedConstellation(
            num_bits_per_symbol=num_bits_per_symbol,
            constellation_type=const_type,
            trainable=False  # 先不启用训练
        )
        constellations[const_type] = constellation
        
        # 可视化星座图
        constellation.visualize(
            title=f'{const_type.upper()} 星座图 ({2**num_bits_per_symbol}点)',
            save_path=f'constellation_{const_type}.png'
        )
    
    # 创建端到端系统进行性能对比
    systems = {}
    
    print("\n创建端到端通信系统...")
    
    for const_type in constellation_types:
        system = EndToEndSystem(
            num_bits_per_symbol=num_bits_per_symbol,
            constellation_type=const_type,
            trainable_constellation=False  # 先不启用训练
        )
        # 构建系统
        system.build()
        systems[const_type] = system
    
    # 测试性能
    ebno_dbs = np.linspace(10.0, 20.0, 6)
    bers = {}
    
    print("\n测试不同星座图的BER性能...")
    
    for const_type, system in systems.items():
        print(f"\n测试 {const_type.upper()} 星座...")
        ber_values = system.test_ber(ebno_dbs, batch_size=1000, num_target_block_errors=50)
        bers[const_type] = ber_values
    
    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    
    colors = {'qam': 'blue', 'psk': 'red', 'custom': 'green'}
    markers = {'qam': 'o', 'psk': 's', 'custom': '^'}
    
    for const_type in constellation_types:
        plt.semilogy(ebno_dbs, bers[const_type], 
                     marker=markers[const_type], 
                     linestyle='-', 
                     color=colors[const_type], 
                     label=f'{const_type.upper()} 星座')
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('误比特率 (BER)')
    plt.title(f'{2**num_bits_per_symbol}-点星座图性能对比')
    plt.legend()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n星座图对比完成！")
    print("生成的文件：")
    for const_type in constellation_types:
        print(f"- constellation_{const_type}.png: {const_type.upper()}星座图")
    print("- constellation_comparison.png: 星座图性能对比图")

# 运行星座图训练和评估
def run_constellation_training(num_bits_per_symbol=6, epochs=10, steps_per_epoch=100):
    """运行星座图训练和评估"""
    # 创建端到端通信系统
    system = EndToEndSystem(
        num_bits_per_symbol=num_bits_per_symbol,
        constellation_type='custom',
        trainable_constellation=True
    )
    
    # 构建系统
    system.build()
    
    # 设置优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    system.compile(optimizer=optimizer)
    
    # 设置训练参数
    system.set_training_params(epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    # 创建基准系统用于对比
    baseline_system = EndToEndSystem(
        num_bits_per_symbol=num_bits_per_symbol,
        constellation_type='qam',
        trainable_constellation=False
    )
    baseline_system.build()
    
    # 可视化初始星座图
    system.visualize_constellation(title='初始星座图', save_path='initial_constellation.png')
    
    # 训练前测试性能
    print("\n训练前测试性能...")
    ebno_dbs = np.linspace(10.0, 20.0, 6)
    initial_bers = system.test_ber(ebno_dbs, batch_size=1000, num_target_block_errors=50)
    
    # 记录训练损失
    history = {'loss': [], 'ber': []}
    
    # 开始训练
    print(f"\n开始训练星座图 ({epochs}轮, 每轮{steps_per_epoch}步)...")
    
    for epoch in range(epochs):
        print(f"\n轮次 {epoch+1}/{epochs}")
        epoch_loss = 0.0
        epoch_ber = 0.0
        
        # 使用tqdm显示进度
        with tqdm(total=steps_per_epoch, desc="  训练进度") as pbar:
            for step in range(steps_per_epoch):
                # 运行一步训练
                # 注意：在实际应用中，应该使用数据生成器
                # 这里为了简化，我们生成随机数据
                batch_size = 256
                num_symbols = 100
                bits = system.binary_source([batch_size, num_symbols * num_bits_per_symbol])
                
                # 计算噪声功率（使用课程学习策略）
                min_ebno = 5.0
                max_ebno = 20.0
                if epoch < epochs * 0.5:
                    ebno_db = min_ebno + (max_ebno - min_ebno) * (epoch / (epochs * 0.5))
                else:
                    ebno_db = max_ebno
                
                # 前向传播和反向传播
                with tf.GradientTape() as tape:
                    llr, loss = system.call(bits, ebno_db=ebno_db)
                    
                    # 添加正则化损失
                    loss += sum(system.losses)
                
                # 计算梯度
                gradients = tape.gradient(loss, system.trainable_variables)
                
                # 应用梯度
                optimizer.apply_gradients(zip(gradients, system.trainable_variables))
                
                # 计算BER
                ber = sn.phy.utils.ber(bits, llr > 0)
                
                # 更新统计
                epoch_loss += loss.numpy()
                epoch_ber += ber.numpy()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.numpy():.4f}', 'ber': f'{ber.numpy():.6f}'})
                pbar.update(1)
        
        # 计算平均损失和BER
        avg_epoch_loss = epoch_loss / steps_per_epoch
        avg_epoch_ber = epoch_ber / steps_per_epoch
        
        # 记录历史
        history['loss'].append(avg_epoch_loss)
        history['ber'].append(avg_epoch_ber)
        
        print(f"  平均损失: {avg_epoch_loss:.4f}")
        print(f"  平均BER: {avg_epoch_ber:.6f}")
    
    # 可视化训练后的星座图
    system.visualize_constellation(title='训练后星座图', save_path='trained_constellation.png')
    
    # 训练后测试性能
    print("\n训练后测试性能...")
    trained_bers = system.test_ber(ebno_dbs, batch_size=1000, num_target_block_errors=100)
    
    # 测试基准系统性能
    print("\n测试基准QAM星座性能...")
    baseline_bers = baseline_system.test_ber(ebno_dbs, batch_size=1000, num_target_block_errors=100)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), history['loss'], marker='o', linestyle='-', color='blue')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练损失曲线')
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制BER性能对比图
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, trained_bers, marker='o', linestyle='-', color='blue', label='训练后星座')
    plt.semilogy(ebno_dbs, initial_bers, marker='s', linestyle='--', color='green', label='初始星座')
    plt.semilogy(ebno_dbs, baseline_bers, marker='^', linestyle='-.', color='red', label='标准QAM')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('误比特率 (BER)')
    plt.title(f'{2**num_bits_per_symbol}-点星座图BER性能对比')
    plt.legend()
    plt.savefig('ber_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n星座图训练和评估完成！")
    print("生成的文件：")
    print("- initial_constellation.png: 初始星座图")
    print("- trained_constellation.png: 训练后星座图")
    print("- training_loss.png: 训练损失曲线")
    print("- ber_comparison.png: BER性能对比图")
    
    # 保存模型权重
    model_path = f'trained_constellation_model.h5'
    system.save_weights(model_path)
    print(f"\n模型权重已保存为: {model_path}")

# 主函数
def main():
    print("高级Sionna星座图优化工具")
    print("=" * 50)
    
    # 设置参数
    NUM_BITS_PER_SYMBOL = 6  # 64-QAM
    
    # 选项菜单
    print("请选择操作:")
    print("1. 对比不同类型的星座图")
    print("2. 运行星座图训练和评估")
    print("3. 退出")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == '1':
        # 对比不同类型的星座图
        compare_constellations(num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
    elif choice == '2':
        # 运行星座图训练和评估
        print("请输入训练参数:")
        try:
            epochs = int(input("轮次 (默认: 10): ") or "10")
            steps_per_epoch = int(input("每轮步数 (默认: 100): ") or "100")
        except ValueError:
            print("输入无效，使用默认参数")
            epochs = 10
            steps_per_epoch = 100
        
        run_constellation_training(
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
    elif choice == '3':
        print("程序已退出")
        sys.exit(0)
    else:
        print("输入无效，请重新运行程序")
        sys.exit(1)

# 运行主函数
if __name__ == "__main__":
    main()