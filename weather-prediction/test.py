import torch

checkpoint = torch.load('weather-prediction/models/trained_model.pth', map_location='cpu')

print(checkpoint.keys())