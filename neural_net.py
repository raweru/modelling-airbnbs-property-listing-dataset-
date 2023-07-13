from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import Dataset
import torch.nn as nn
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

# Split the dataset into training and validation sets
X, y = dataset.X, dataset.y
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the training dataset
train_dataset = AirbnbNightlyPriceRegressionDataset()
train_dataset.X, train_dataset.y = X_train, y_train

# Create the validation dataset
val_dataset = AirbnbNightlyPriceRegressionDataset()
val_dataset.X, val_dataset.y = X_val, y_val

# Create the data loaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # initalise the parameters
        self.linear_layer = nn.Linear(11, 1)

    def forward(self, features):
        return self.linear_layer(features)  # make prediction


model = LinearRegression()

def train(model, epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    
    writer = SummaryWriter()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Training loop
        for features, labels in train_loader:
            features = features.to(torch.float32)
            labels = labels.to(torch.float32)
            prediction = model(features)
            if epoch == 0:
                print(prediction)
            loss = loss_function(prediction, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # # Validation loop
        with torch.no_grad():
            running_val_loss = 0.0
            
            for features, labels in val_loader:
                features = features.to(torch.float32)
                labels = labels.to(torch.float32)
                prediction = model(features)
                loss = loss_function(prediction, labels)
                running_val_loss += loss.item()
            
            # Calculate average validation loss for the epoch
            avg_val_loss = running_val_loss / len(val_loader)
        
        # Print epoch information
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
    # Close the SummaryWriter
    writer.close()


def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(torch.float32)
            targets = targets.to(torch.float32)
            
            # Make predictions
            output = model(features)
            
            # Convert predictions and targets to numpy arrays
            predictions.extend(output.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    
    # Calculate R2 score
    r2 = r2_score(labels, predictions)
    
    return rmse, r2


train(model)

# Evaluate the model on the training set
train_rmse, train_r2 = evaluate_model(model, train_loader)
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Train R2: {train_r2:.4f}")

# Evaluate the model on the validation set
val_rmse, val_r2 = evaluate_model(model, val_loader)
print(f"Val RMSE: {val_rmse:.4f}")
print(f"Val R2: {val_r2:.4f}")
