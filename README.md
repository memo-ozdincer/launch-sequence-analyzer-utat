# Launch Sequence Analyzer

Neural network-based system for analyzing engine test data and detecting anomalies in real-time. Essentially a test system for some of the telemetry stuff we've been working on, kind of for fun, but maybe we will use it.

## Overview

- Processes temperature, pressure, and vibration sensor data at 10Hz
- Predicts anomalies using PyTorch neural network
- Provides real-time visualization and alerts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train model:
```python
python train_model.py --data path/to/your/data.csv
```

Run monitor:
```python
python monitor.py
```

## Technical Details

- PyTorch neural network classifier
- 500ms detection window (10 samples)
- 89% accuracy on test data
- Pandas/NumPy for data processing
- Threaded sensor reading
- Matplotlib for visualization

## Future Work

- LSTM implementation
- Additional sensor inputs
- Web interface

## Contact

memo.ozdincer@mail.utoronto.ca
