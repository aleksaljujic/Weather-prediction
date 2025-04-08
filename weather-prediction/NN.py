import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib as jb

def load_data(train_path, test_path):

    #read the data in csv format
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    #slect thrain columns
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    #select test columns
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    #standardize(normal distribution) the data
    X_scaler = StandardScaler()   
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    # Save scalers
    '''
    jb.dump(X_scaler, 'models/X_scaler.pkl')
    jb.dump(y_scaler, 'models/y_scaler.pkl')
    '''

    #convert the data to tensor format
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) 

    #create train loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    #create test loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)   
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

def train_model(train_path, test_path, num_epochs=10, lr=0.001):
    
    train_loader, test_loader = load_data(train_path, test_path)

    input_size = train_loader.dataset[0][0].shape[0] 
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    #save_model(model, 'models/trained_model.pth')
    evaluate_model(model, test_loader) 

def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        avg_loss = total_loss / (len(test_loader)*32)
        print(f'Average Loss: {avg_loss:.4f}')

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print("Model saved successfully")

def load_model(file_path, input_size):
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

def predict(model, X_scaler, y_scaler):
    X_scaler=jb.load(X_scaler)
    y_scaler=jb.load(y_scaler)

    input_size = X_scaler.n_features_in_
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(model, map_location='cpu'))
    model.eval()

    custom_input = list(map(float, input("Enter the input values (comma-separated): ").split(',')))

    scaled_input = X_scaler.transform([custom_input])
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    #predict
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction_original = y_scaler.inverse_transform(prediction.numpy())
        print(f'Predicted Output: {prediction_original[0][0]:.4f}')

if __name__ == "__main__":

    train_model(
        train_path='data/train_data.csv',  
        test_path='data/test_data.csv',    
        num_epochs=100,               
        lr=0.001                      
    )
    '''
    predict(
        model_path='models/trained_model.pth',
        X_scaler_path='models/X_scaler.pkl',
        y_scaler_path='models/y_scaler.pkl'
    )
    '''