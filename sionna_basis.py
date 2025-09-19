import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
# %matplotlib inline
# also try %matplotlib widget

import matplotlib.pyplot as plt

# for performance measurements
import time

# !nvidia-smi

sionna.phy.config.seed = 40

# Python RNG - use instead of
# import random
# random.randint(0, 10)
print(sionna.phy.config.py_rng.randint(0,10))

# NumPy RNG - use instead of
# import numpy as np
# np.random.randint(0, 10)
print(sionna.phy.config.np_rng.integers(0,10))

# TensorFlow RNG - use instead of
# import tensorflow as tf
# tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32)
print(sionna.phy.config.tf_rng.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32))


NUM_BITS_PER_SYMBOL = 4 # QPSK
constellation = sionna.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

# constellation.show();
# plt.show()


mapper = sionna.phy.mapping.Mapper(constellation=constellation)

# The demapper uses the same constellation object as the mapper
demapper = sionna.phy.mapping.Demapper("app", constellation=constellation)
# demapper = sionna.phy.mapping.Demapper("maxlog", constellation=constellation)


# print class definition of the Constellation class
# sionna.phy.mapping.Mapper??


binary_source = sionna.phy.mapping.BinarySource()

awgn_channel = sionna.phy.channel.AWGN()

no = sionna.phy.utils.ebnodb2no(ebno_db=20.0,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here

BATCH_SIZE = 64 # How many examples are processed by Sionna in parallel

bits = binary_source([BATCH_SIZE,
                      1024]) # Blocklength
print("Shape of bits: ", bits.shape)

x = mapper(bits)
print("Shape of x: ", x.shape)

y = awgn_channel(x, no)
print("Shape of y: ", y.shape)

llr = demapper(y, no)
print("Shape of llr: ", llr.shape)


num_samples = 8 # how many samples shall be printed
num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)

# print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
# print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
# print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
# print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}")

# plt.figure(figsize=(8,8))
# plt.axes().set_aspect(1)
# plt.grid(True)
# plt.title('Channel output')
# plt.xlabel('Real Part')
# plt.ylabel('Imaginary Part')
# plt.scatter(tf.math.real(y), tf.math.imag(y))
# plt.tight_layout()
# plt.show()

# Method 1: Plot LLR values of a single batch sample (choose the first sample)
plt.figure(figsize=(12, 6))
plt.plot(llr[0, :], '.', markersize=4)
plt.title('LLR Values of the First Sample')
plt.xlabel('Bit Position')
plt.ylabel('LLR Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Method 2: Plot LLR values of multiple batch samples (choose first 5 samples)
plt.figure(figsize=(12, 8))
for i in range(min(5, BATCH_SIZE)):
    plt.plot(llr[i, :], '.', markersize=3, label=f'Sample {i+1}')
plt.title('LLR Values of Multiple Samples')
plt.xlabel('Bit Position')
plt.ylabel('LLR Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Method 3: Plot heatmap of LLR values (visualize all samples)
plt.figure(figsize=(12, 8))
# Convert to numpy array and transpose for better visualization
llr_np = llr.numpy().T
plt.imshow(llr_np, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='LLR Value')
plt.title('Heatmap of LLR Values')
plt.xlabel('Sample Index')
plt.ylabel('Bit Position')
plt.tight_layout()
plt.show()

# Method 4: Plot histogram of LLR values
plt.figure(figsize=(10, 6))
plt.hist(llr.numpy().flatten(), bins=50, alpha=0.7, color='blue')
plt.title('Histogram of LLR Value Distribution')
plt.xlabel('LLR Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()