# Delhi Weather Prediction Neural Network

A PyTorch-based neural network implementation for predicting weather conditions in Delhi using historical meteorological data.

## Project Overview

This project implements a deep learning model to forecast weather conditions, specifically mean temperature, based on various meteorological features from Delhi's historical weather data.

### Key Features

- Neural network-based weather prediction
- PyTorch implementation with batch normalization
- Comprehensive data preprocessing pipeline
- Model training and evaluation functionality
- Support for both model training and inference

## Technical Architecture

### Input Features
- Humidity
- Wind speed
- Mean pressure
- Temperature differences (temporal)
- Weather metric differences (temporal)

### Model Architecture
- 4-layer neural network
- Batch normalization layers
- ReLU activation functions
- MSE loss function
- Adam optimizer

## Project Structure

```
weather-prediction/
├── NN.py                  # Core neural network implementation
├── data/                  # Dataset directory
│   ├── train_data.csv     # Training dataset
│   └── test_data.csv      # Test dataset
├── notebooks/             # Jupyter notebooks
│   └── data_preprocessing.ipynb
└── requirements.txt       # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/aleksaljujic/Weather-prediction.git
cd Weather-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
from NN import train_model

train_model(
    train_path='data/train_data.csv',
    test_path='data/test_data.csv',
    num_epochs=100,
    lr=0.001
)
```

### Making Predictions

```python
from NN import predict

predict(
    model_path='models/trained_model.pth',
    X_scaler_path='models/X_scaler.pkl',
    y_scaler_path='models/y_scaler.pkl'
)
```

## Data Preprocessing

The project includes a comprehensive data preprocessing pipeline that handles:
- Data normalization using StandardScaler
- Feature engineering
- Train/test data splitting
- Data validation and cleaning

## Model Performance

The model uses MSE (Mean Squared Error) as its loss function and includes:
- Batch normalization for improved training stability
- Gradient clipping to prevent exploding gradients
- Model evaluation on test dataset

## Dependencies

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter (for notebooks)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
