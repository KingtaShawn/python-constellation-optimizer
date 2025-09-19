# Advanced Constellation Optimization with Improved Techniques

import os
import tensorflow as tf
import sionna as sn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers

# Environment Setup
def setup_environment():
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sn.phy.config.seed = 42
    np.random.seed(42)
    tf.random.set_seed(42)

# Custom Constellation Class with Relaxed Constraints
class CustomConstellation(tf.keras.layers.Layer):
    def __init__(self, num_bits_per_symbol, initial_points=None, trainable=True):
        super(CustomConstellation, self).__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_points = 2 ** num_bits_per_symbol
        
        if initial_points is None:
            # Create initial QAM constellation
            qam = sn.phy.mapping.Constellation("qam", num_bits_per_symbol)
            initial_points = tf.stack([tf.math.real(qam.points), 
                                       tf.math.imag(qam.points)], axis=0)
        
        # Make points trainable
        self.points = tf.Variable(initial_points, dtype=tf.float32, trainable=trainable)
        
    def call(self):
        # Return complex constellation points without strict normalization
        complex_points = tf.complex(self.points[0], self.points[1])
        
        # Optional: Soft power normalization (weaker constraint)
        power = tf.reduce_mean(tf.square(tf.abs(complex_points)))
        normalized_points = complex_points / tf.sqrt(power)
        
        return normalized_points

# Custom Loss Function with Distance Regularization
def custom_loss_function(bits, llr, constellation_points, distance_weight=0.1):
    # Standard BCE loss
    bce_loss = tf.keras.losses.binary_crossentropy(bits, llr, from_logits=True)
    
    # Distance regularization: Encourage minimum distance between points
    # Calculate pairwise distances between all constellation points
    complex_points = tf.complex(constellation_points[0], constellation_points[1])
    diffs = complex_points[:, tf.newaxis] - complex_points[tf.newaxis, :]
    distances = tf.abs(diffs)
    
    # Exclude self-distances (diagonal elements)
    min_distance = tf.reduce_min(tf.boolean_mask(distances, tf.eye(len(complex_points)) < 0.5))
    
    # Regularization term: Penalize small minimum distances
    distance_reg = distance_weight * (1.0 / (min_distance + 1e-10))
    
    # Total loss
    total_loss = bce_loss + distance_reg
    
    return total_loss

# Curriculum Learning Scheduler
class CurriculumScheduler:
    def __init__(self, min_snr, max_snr, total_epochs, ramp_up_epochs=200):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.total_epochs = total_epochs
        self.ramp_up_epochs = ramp_up_epochs
    
    def get_snr(self, epoch):
        # Linear ramp up from min_snr to max_snr during ramp_up_epochs
        if epoch < self.ramp_up_epochs:
            snr = self.min_snr + (self.max_snr - self.min_snr) * (epoch / self.ramp_up_epochs)
        else:
            # After ramp-up, use a random SNR between min and max
            snr = self.min_snr + np.random.random() * (self.max_snr - self.min_snr)
        return snr

# Main Training Function with Advanced Techniques
def train_advanced_constellation():
    setup_environment()
    
    # Hyperparameters
    NUM_BITS_PER_SYMBOL = 6
    BATCH_SIZE = 256  # Larger batch size for stability
    LEARNING_RATE = 5e-4  # Smaller learning rate
    EPOCHS = 2000  # More epochs
    PATIENCE = 100  # Longer patience for early stopping
    
    # Curriculum learning parameters
    MIN_SNR = 12.0
    MAX_SNR = 20.0
    curriculum = CurriculumScheduler(MIN_SNR, MAX_SNR, EPOCHS)
    
    # Modules
    binary_source = sn.phy.mapping.BinarySource()
    awgn_channel = sn.phy.channel.AWGN()
    
    # Create custom trainable constellation
    constellation = CustomConstellation(NUM_BITS_PER_SYMBOL)
    
    # Optimizer with learning rate decay
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=5000, decay_rate=0.9, staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=scheduler)
    
    # Training loop with curriculum learning
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        current_snr = curriculum.get_snr(epoch)
        no = sn.phy.utils.ebnodb2no(
            ebno_db=current_snr,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=1.0
        )
        
        with tf.GradientTape() as tape:
            bits = binary_source([BATCH_SIZE, 1200])
            
            # Get current constellation points
            complex_points = constellation()
            
            # Create mapper and demapper
            qam = sn.phy.mapping.Constellation("custom", NUM_BITS_PER_SYMBOL, complex_points)
            mapper = sn.phy.mapping.Mapper(constellation=qam)
            demapper = sn.phy.mapping.Demapper("app", constellation=qam)
            
            # Forward pass
            x = mapper(bits)
            y = awgn_channel(x, no)
            llr = demapper(y, no)
            
            # Custom loss with distance regularization
            loss = custom_loss_function(bits, llr, constellation.points)
        
        # Backward pass and optimization
        gradients = tape.gradient(loss, constellation.trainable_variables)
        optimizer.apply_gradients(zip(gradients, constellation.trainable_variables))
        
        # Track loss and implement early stopping
        loss_history.append(loss.numpy())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | SNR: {current_snr:.1f} dB | Loss: {loss.numpy():.6f}")
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save the best model
            tf.saved_model.save(constellation, 'best_constellation')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return constellation, loss_history

# Run the advanced training if this script is executed directly
if __name__ == "__main__":
    print("Starting advanced constellation optimization...")
    constellation, loss_history = train_advanced_constellation()
    
    # Visualize training progress
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('advanced_training_loss.png')
    plt.close()
    
    print("Advanced optimization completed!")