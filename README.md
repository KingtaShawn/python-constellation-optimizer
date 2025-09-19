# Python Constellation Optimizer

This project implements constellation optimization techniques using the Sionna framework for communication systems.

## Features
- Constellation optimization with deep learning
- Bit Error Rate (BER) performance analysis
- Visualization of constellation diagrams
- Comparison between baseline and optimized constellations

## Key Files
- `simple_constellation_optimizer.py`: Main implementation for constellation optimization
- `advanced_sionna_constellation.py`: Advanced implementation with custom layers
- `fixed_advanced_constellation.py`: Fixed version of advanced implementation
- Various visualization files (PNG format)

## Requirements
- Python 3.x
- TensorFlow/Sionna
- NumPy
- Matplotlib

## Usage
Run the main optimization script:
```bash
python simple_constellation_optimizer.py
```

This will generate:
- `constellation_comparison.png`: Comparison of baseline vs optimized constellation
- `ber_comparison.png`: BER performance comparison
- `training_loss.png`: Training loss curve