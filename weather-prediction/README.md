# Weather Prediction Neural Network

This project implements a neural network using PyTorch to predict weather conditions based on historical data. 

## Project Structure

- **data/**: Contains the dataset and related documentation.
  - **README.md**: Documentation related to the dataset used for weather prediction.
  
- **models/**: Contains the neural network architecture.
  - **neural_network.py**: Defines the NeuralNetwork class implementing the architecture using PyTorch.

- **notebooks/**: Jupyter notebooks for data analysis and preprocessing.
  - **data_preprocessing.ipynb**: Notebook for cleaning and preparing the weather data.

- **src/**: Source code for training and prediction.
  - **train.py**: Script for training the neural network.
  - **predict.py**: Script for making predictions with a blank field for user input.
  - **utils.py**: Utility functions for data handling and evaluation.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd weather-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Use the `data_preprocessing.ipynb` notebook to preprocess the dataset before training the model.
- Run `train.py` to train the neural network on the prepared dataset.
- Use `predict.py` to make predictions on new weather data.

## Neural Network Model

The neural network is designed to learn patterns from historical weather data and make accurate predictions. The architecture and training process are defined in the `models/neural_network.py` and `src/train.py` files, respectively.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.