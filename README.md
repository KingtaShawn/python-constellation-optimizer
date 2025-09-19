# Python Constellation Optimization Project

This project implements constellation optimization using the Sionna framework. It demonstrates how to optimize constellation points for better BER performance in AWGN channels compared to traditional QAM constellations.

## Features

- Constellation optimization using TensorFlow and Sionna
- BER performance comparison between traditional QAM and optimized constellations
- Visualization of constellation points, BER curves, and training dynamics
- Multi-SNR training for robust constellation design
- Power constraint regularization
- Early stopping for efficient training

## Key Files

- `simple_constellation_optimizer.py`: Main optimization script with simplified implementation
- `requirements.txt`: Project dependencies
- `upload_to_github.py`: Script to upload visualization files to GitHub
- `upload_images.py`: Alternative script for uploading images

## Requirements

- Python 3.7+
- TensorFlow >= 2.10.0
- Sionna >= 0.15.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage

Run the main optimization script:
```
python simple_constellation_optimizer.py
```

This will:
1. Optimize a 64-QAM constellation using gradient descent
2. Compare BER performance with traditional 64-QAM
3. Generate visualization files:
   - `constellation_comparison.png`: Original vs. optimized constellation
   - `ber_comparison.png`: BER performance comparison
   - `training_loss.png`: Training loss history

## Results

The optimized constellation achieves significantly better BER performance than traditional 64-QAM, especially at higher SNR values. In the range of 12-21dB Eb/No, the BER reduction ranges from 2.77% to 24.10%.

## Visualization

To view the visualization results, you need to run the optimization script locally as GitHub does not display the generated PNG files in the repository. After running the script, you can use the upload scripts to upload the generated visualization files to GitHub if desired.