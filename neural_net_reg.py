from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tabular_data import load_airbnb
import yaml
import os
import time
from datetime import datetime
from regression import save_model
import itertools
import json
import random


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb(label="bedrooms")
    
    def __getitem__(self, idx):
        return (torch.tensor(self.X.iloc[idx].values), torch.tensor(self.y.iloc[idx].values))
    
    def __len__(self):
        return len(self.X)


dataset = AirbnbNightlyPriceRegressionDataset()
batch_size = 8

# Split the dataset into training, validation and test sets
X, y = dataset.X, dataset.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create the training dataset
train_dataset = AirbnbNightlyPriceRegressionDataset()
train_dataset.X, train_dataset.y = X_train, y_train

# Create the validation dataset
val_dataset = AirbnbNightlyPriceRegressionDataset()
val_dataset.X, val_dataset.y = X_val, y_val

# Create the test dataset
test_dataset = AirbnbNightlyPriceRegressionDataset()
test_dataset.X, test_dataset.y = X_test, y_test

# Create the data loaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

def get_nn_config():
    config_path = 'nn_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # initalise the parameters
        self.linear_layer = nn.Linear(12, 1)

    def forward(self, features):
        return self.linear_layer(features)  # make prediction


class MLPRegressor(nn.Module):
    def __init__(self, config):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(12, config["hidden_layer_width"])
        self.fc2 = nn.Linear(config["hidden_layer_width"], 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


config = get_nn_config()
model = MLPRegressor(config)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
        self.loss_function = nn.MSELoss()
        self.writer = SummaryWriter()

    def evaluate_model(self, data_loader):
        n_samples = len(data_loader.dataset)
        start_time = time.time()
        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(torch.float32)
                targets = targets.to(torch.float32)

                # Make predictions
                output = self.model(features)

                # Convert predictions and targets to numpy arrays
                predictions.extend(output.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        end_time = time.time()
        inference_latency = (end_time - start_time) / n_samples
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(labels, predictions))

        # Calculate R2 score
        r2 = r2_score(labels, predictions)

        return rmse, r2, inference_latency

    def train(self):
        start_time = time.time()
        metrics_dict = {
            "train": {"RMSE_loss": None, "R_squared": None, "inference_latency": None},
            "val": {"RMSE_loss": None, "R_squared": None, "inference_latency": None},
            "test": {"RMSE_loss": None, "R_squared": None, "inference_latency": None}
        }

        for epoch in range(self.config["num_epochs"]):
            running_loss = 0.0

            # Training loop
            for features, labels in self.train_loader:
                features = features.to(torch.float32)
                labels = labels.to(torch.float32)
                prediction = self.model(features)
                loss = torch.sqrt(self.loss_function(prediction, labels))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader)

            # Validation loop
            with torch.no_grad():
                running_val_loss = 0.0

                for features, labels in self.val_loader:
                    features = features.to(torch.float32)
                    labels = labels.to(torch.float32)
                    prediction = self.model(features)
                    loss = torch.sqrt(self.loss_function(prediction, labels))
                    running_val_loss += loss.item()

                # Calculate average validation loss for the epoch
                avg_val_loss = running_val_loss / len(self.val_loader)

            # Test loop
            with torch.no_grad():
                running_test_loss = 0.0

                for features, labels in self.test_loader:
                    features = features.to(torch.float32)
                    labels = labels.to(torch.float32)
                    prediction = self.model(features)
                    loss = torch.sqrt(self.loss_function(prediction, labels))
                    running_test_loss += loss.item()

                # Calculate average test loss for the epoch
                avg_test_loss = running_test_loss / len(self.test_loader)

            # Print epoch information
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

            # Save the metrics for the last epoch only
            if epoch == self.config['num_epochs'] - 1:
                metrics_dict["train"]["RMSE_loss"] = avg_train_loss
                metrics_dict["train"]["R_squared"] = self.evaluate_model(self.train_loader)[1]
                metrics_dict["train"]["inference_latency"] = self.evaluate_model(self.train_loader)[2]
                metrics_dict["val"]["RMSE_loss"] = avg_val_loss
                metrics_dict["val"]["R_squared"] = self.evaluate_model(self.val_loader)[1]
                metrics_dict["val"]["inference_latency"] = self.evaluate_model(self.val_loader)[2]
                metrics_dict["test"]["RMSE_loss"] = avg_test_loss
                metrics_dict["test"]["R_squared"] = self.evaluate_model(self.test_loader)[1]
                metrics_dict["test"]["inference_latency"] = self.evaluate_model(self.test_loader)[2]

            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        end_time = time.time()
        training_duration = end_time - start_time
        self.training_duration = training_duration
        self.writer.close()

        return metrics_dict


# Train the model and save metrics
trainer = Trainer(model, train_loader, val_loader, test_loader, config)
metrics_dict = trainer.train()

# Add training duration to the metrics dictionary
metrics_dict["training_duration"] = trainer.training_duration

# Save the model and metrics in a new folder based on current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join("models", "neural_networks", "regression", current_time)
hyperparameters = {
    "learning_rate": config["learning_rate"],
    "hidden_layer_width": config["hidden_layer_width"],
    "num_epochs": config["num_epochs"]
}


def generate_nn_configs():
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [25, 50, 100]
    num_epochs = [100, 200, 500]

    # Generate all combinations of hyperparameters
    configs = list(itertools.product(learning_rates, hidden_layer_widths, num_epochs))

    # Shuffle the configurations randomly
    random.shuffle(configs)

    # Select up to 16 configurations
    max_configs = 16
    if len(configs) > max_configs:
        configs = configs[:max_configs]

    nn_configs = []
    for config in configs:
        nn_config = {
            "learning_rate": config[0],
            "hidden_layer_width": config[1],
            "num_epochs": config[2]
        }
        nn_configs.append(nn_config)

    return nn_configs



def find_best_nn(train_loader, val_loader, test_loader):
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_val_loss = float('inf')

    configs = generate_nn_configs()
    for i, config in enumerate(configs):
        print(f"Training model {i + 1} out of {len(configs)}...")
        model = MLPRegressor(config)
        trainer = Trainer(model, train_loader, val_loader, test_loader, config)
        metrics_dict = trainer.train()

        val_loss = metrics_dict["val"]["RMSE_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_metrics = metrics_dict
            best_hyperparameters = config

        # Stop after training 16 models
        if i == 15:
            break

    return best_model, best_metrics, best_hyperparameters


# Assuming you already have the train_loader, val_loader, and test_loader defined
best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader, val_loader, test_loader)

# Save the best model, its metrics, and hyperparameters
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join("models", "neural_networks", "regression", current_time)
os.makedirs(folder_path, exist_ok=True)

# Save the model
model_path = os.path.join(folder_path, "best_model.pt")
torch.save(best_model.state_dict(), model_path)

# Save the hyperparameters
hyperparameters_path = os.path.join(folder_path, "hyperparameters.json")
with open(hyperparameters_path, "w") as file:
    json.dump(best_hyperparameters, file)

# Save the metrics
metrics_path = os.path.join(folder_path, "metrics.json")
with open(metrics_path, "w") as file:
    json.dump(best_metrics, file)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Metrics:", best_metrics)
