from typing import Any
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tabular_data import load_airbnb


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb(label="Price_Night")
    
    def __getitem__(self, idx):
        return (torch.tensor(self.X.iloc[idx].values), torch.tensor(self.y.iloc[idx].values))
    
    def __len__(self):
        return len(self.X)


dataset = AirbnbNightlyPriceRegressionDataset()
batch_size = 4
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

example = next(iter(train_loader))
features, labels = example

class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # initalise the parameters
        self.linear_layer = torch.nn.Linear(11, 1)
    
    def forward(self, features):
        return self.linear_layer(features)


model = LinearRegression()
features = features.to(model.linear_layer.weight.dtype)
print(model(features))