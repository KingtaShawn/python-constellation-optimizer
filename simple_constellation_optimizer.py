# Simplified Constellation Optimization with Reliable Implementation

import os
import tensorflow as tf
import sionna as sn
import numpy as np
import matplotlib.pyplot as plt

# Environment Setup
def setup_environment():
    # Configure logging and random seeds for reproducibility
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sn.phy.config.seed = 42
    np.random.seed(42)
    tf.random.set_seed(42)

# Main Training Function with Simplified Implementation
def train_simplified_constellation():
    setup_environment()
    
    # Hyperparameters
    NUM_BITS_PER_SYMBOL = 6
    NUM_SYMBOLS = 2 ** NUM_BITS_PER_SYMBOL
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 1000
    PATIENCE = 50
    
    # Create initial QAM constellation
    qam = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    initial_points = qam.points.numpy()
    
    # Make points trainable (using complex values directly)
    # This is simpler and more reliable than splitting into real/imaginary parts
    trainable_points = tf.Variable(initial_points, dtype=tf.complex64, trainable=True)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Modules
    binary_source = sn.phy.mapping.BinarySource()
    awgn_channel = sn.phy.channel.AWGN()
    
    # Training loop with early stopping
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    best_points = tf.identity(trainable_points)
    
    # Use multiple SNRs during training for robustness
    train_ebno_dbs = np.linspace(12.0, 18.0, 5)
    
    for epoch in range(EPOCHS):
        # Randomly select an SNR for this epoch
        current_ebno_db = np.random.choice(train_ebno_dbs)
        no = sn.phy.utils.ebnodb2no(
            ebno_db=current_ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=1.0
        )
        
        with tf.GradientTape() as tape:
            # Track the trainable constellation points
            tape.watch([trainable_points])
            
            # Create constellation and mapper/demapper
            qam = sn.phy.mapping.Constellation("custom", NUM_BITS_PER_SYMBOL, trainable_points)
            mapper = sn.phy.mapping.Mapper(constellation=qam)
            demapper = sn.phy.mapping.Demapper("app", constellation=qam)
            
            # Generate data and perform modulation
            bits = binary_source([BATCH_SIZE, 1200])
            x = mapper(bits)
            
            # Transmit over AWGN channel
            y = awgn_channel(x, no)
            
            # Demodulate and compute loss
            llr = demapper(y, no)
            loss = tf.keras.losses.binary_crossentropy(bits, llr, from_logits=True)
            
            # Add power constraint regularization
            power = tf.reduce_mean(tf.square(tf.abs(trainable_points)))
            power_reg = 0.01 * tf.square(power - 1.0)  # Target power is 1
            loss = loss + power_reg
        
        # Compute gradients and apply updates
        gradients = tape.gradient(loss, [trainable_points])
        optimizer.apply_gradients(zip(gradients, [trainable_points]))
        
        # Track loss
        loss_value = float(tf.reduce_mean(loss))
        loss_history.append(loss_value)
        
        # Early stopping
        if loss_value < best_loss:
            best_loss = loss_value
            patience_counter = 0
            best_points = tf.identity(trainable_points)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} with best loss: {best_loss:.6f}")
                break
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value:.6f}, SNR: {current_ebno_db:.2f} dB")
    
    return best_points, loss_history

# Simulation Function to Evaluate BER
def simulate_ber(constellation_points, ebno_dbs, num_bits=1000000):
    # Create constellation, mapper and demapper
    num_bits_per_symbol = int(np.log2(len(constellation_points)))
    qam = sn.phy.mapping.Constellation("custom", num_bits_per_symbol, constellation_points)
    mapper = sn.phy.mapping.Mapper(constellation=qam)
    demapper = sn.phy.mapping.Demapper("app", constellation=qam)
    
    # Binary source and channel
    binary_source = sn.phy.mapping.BinarySource()
    awgn_channel = sn.phy.channel.AWGN()
    
    # Simulate for each Eb/No
    ber = np.zeros_like(ebno_dbs, dtype=np.float64)
    
    for i, ebno_db in enumerate(ebno_dbs):
        no = sn.phy.utils.ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=num_bits_per_symbol,
            coderate=1.0
        )
        
        # Simulate in batches to avoid memory issues
        bits_errors = 0
        total_bits = 0
        
        while total_bits < num_bits:
            # Adjust batch size for last iteration
            current_batch_size = min(10000, num_bits - total_bits)
            
            # Generate bits and modulate
            bits = binary_source([current_batch_size, 1200])
            x = mapper(bits)
            
            # Transmit over AWGN channel
            y = awgn_channel(x, no)
            
            # Demodulate
            llr = demapper(y, no)
            
            # Hard decision and count errors
            bits_hat = np.round(1.0 / (1.0 + np.exp(-llr))).astype(int)
            bits_errors += np.sum(bits != bits_hat)
            total_bits += np.prod(bits.shape)
        
        ber[i] = bits_errors / total_bits
        print(f"Eb/No: {ebno_db:.2f} dB, BER: {ber[i]:.6e}")
    
    return ber

# Visualization Functions
def plot_constellation_comparison(original_points, optimized_points):
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Original 64QAM
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(original_points), np.imag(original_points), color='blue', s=50)
    plt.title('Original 64QAM Constellation')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.axis('equal')
    
    # Optimized Constellation
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(optimized_points), np.imag(optimized_points), color='red', s=50)
    plt.title('SGD-Optimized Constellation')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('constellation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ber_comparison(ebno_dbs, baseline_ber, optimized_ber):
    plt.figure(figsize=(10, 8))
    plt.semilogy(ebno_dbs, baseline_ber, 'b-o', linewidth=2, markersize=8, label='Traditional 64QAM (Baseline)')
    plt.semilogy(ebno_dbs, optimized_ber, 'r-^', linewidth=2, markersize=8, label='SGD-Optimized 64QAM')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.title('AWGN Channel: Traditional 64QAM vs. SGD-Optimized 64QAM', fontsize=14)
    plt.xlabel('Signal-to-Noise Ratio Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim([min(ebno_dbs), max(ebno_dbs)])
    plt.ylim([1e-6, 1e-1])
    plt.savefig('ber_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'k-', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Training Loss History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main Execution
if __name__ == "__main__":
    print("Starting simplified constellation optimization...")
    
    # Train optimized constellation
    optimized_points, loss_history = train_simplified_constellation()
    
    # Create baseline QAM constellation for comparison
    qam = sn.phy.mapping.Constellation("qam", 6)
    baseline_points = qam.points.numpy()
    
    # Simulate BER for both constellations
    ebno_dbs = np.arange(12, 22, 1)
    print("Simulating BER for baseline constellation...")
    baseline_ber = simulate_ber(baseline_points, ebno_dbs)
    print("Simulating BER for optimized constellation...")
    optimized_ber = simulate_ber(optimized_points, ebno_dbs)
    
    # Calculate BER reduction percentage
    ber_reduction = ((baseline_ber - optimized_ber) / baseline_ber) * 100
    
    # Find the maximum SNR gain
    snr_gain = np.zeros_like(baseline_ber)
    for i in range(len(baseline_ber)):
        if baseline_ber[i] > 1e-6:
            # Find the index where optimized BER is closest to baseline BER[i]
            closest_idx = np.argmin(np.abs(optimized_ber - baseline_ber[i]))
            if closest_idx < len(ebno_dbs) - 1:
                snr_gain[i] = ebno_dbs[i] - ebno_dbs[closest_idx]
    
    max_snr_gain = np.max(snr_gain)
    
    # Print results
    print("\n--- Performance Results ---")
    for i in range(len(ebno_dbs)):
        print(f"Eb/No: {ebno_dbs[i]:.2f} dB, Baseline BER: {baseline_ber[i]:.6e}, "
              f"Optimized BER: {optimized_ber[i]:.6e}, BER Reduction: {ber_reduction[i]:.2f}%")
    
    print(f"\nMaximum SNR Gain: {max_snr_gain:.2f} dB")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_constellation_comparison(baseline_points, optimized_points)
    plot_ber_comparison(ebno_dbs, baseline_ber, optimized_ber)
    plot_loss_history(loss_history)
    
    print("\nAnalysis completed successfully!\n" 
          "Generated files:\n" 
          "- constellation_comparison.png: Original vs. Optimized Constellation\n" 
          "- ber_comparison.png: BER Performance Comparison\n" 
          "- training_loss.png: Training Loss History")