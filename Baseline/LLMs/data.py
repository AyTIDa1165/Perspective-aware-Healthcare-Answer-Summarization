import json
import torch
from torch.utils.data import Dataset, DataLoader

class CustomJSONDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Path to your JSON file
file_path = 'path_to_your_file.json'

# Initialize Dataset and DataLoader
dataset = CustomJSONDataset(file_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get the first data point
first_data_point = next(iter(dataloader))
print(first_data_point)
