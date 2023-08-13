import itertools
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import yaml
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from regression import save_model
from tabular_data import load_airbnb


class AirbnbNightlyPriceRegressionDataset(Dataset):
    
    '''
    The `AirbnbNightlyPriceRegressionDataset` class is a PyTorch dataset that loads Airbnb data and
    returns input-output pairs for regression tasks.
    '''
    
    def __init__(self):
        
        '''
        The function initializes the X and y variables by loading the Airbnb data and converting them to
        torch tensors.
        '''
        
        super().__init__()
        self.X, self.y = load_airbnb(label="Price_Night")
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    
    def __getitem__(self, idx):
        
        '''
        The `__getitem__` function returns a tuple containing the `X` and `y` values at the specified
        index.
        
        Parameters
        ----------
        idx
            The parameter "idx" is the index of the item that you want to retrieve from the object. It is
        used to access the corresponding elements in the "X" and "y" attributes of the object.
        
        Returns
        -------
            The code is returning a tuple containing the elements at index `idx` from the `self.X` and
        `self.y` lists.
        '''
        
        return (self.X[idx], self.y[idx])
    
    def __len__(self):
        
        '''
        The function returns the length of the attribute "X" of the object.
        
        Returns
        -------
            The length of the attribute `self.X` is being returned.
        '''
        
        return len(self.X)


def get_nn_config():
    
    '''
    The function `get_nn_config()` reads a YAML file containing neural network configuration and returns
    the configuration as a dictionary.
    
    Returns
    -------
        the content of the `nn_config.yaml` file as a dictionary.
    '''
    
    config_path = 'nn_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    return config


class LinearRegression(nn.Module):
    
    '''
    The LinearRegression class defines a linear regression model with 12 input features and 1 output
    feature.
    '''
    
    def __init__(self):
        
        '''
        The function initializes a linear layer with 12 input features and 1 output feature.
        '''
        
        super().__init__()
        self.linear_layer = nn.Linear(12, 1)

    def forward(self, features):
        
        '''
        The forward function takes in a set of features and passes them through a linear layer.
        
        Parameters
        ----------
        features
            The "features" parameter is a tensor that represents the input features to the forward method.
        It is passed as an argument to the linear_layer method, which applies a linear transformation to
        the features. The output of the linear_layer method is then returned as the result of the
        forward method.
        
        Returns
        -------
            The output of the linear layer applied to the input features.
        '''
        
        return self.linear_layer(features)


class MLPRegressor(nn.Module):
    
    '''
    The MLPRegressor class is a neural network model for regression tasks with configurable hidden layer
    width and depth.
    '''

    def __init__(self, config):
        
        '''
        The `__init__` function initializes an MLPRegressor object with the given configuration,
        including the depth and widths of hidden layers, and creates the necessary layers for the neural
        network.
        
        Parameters
        ----------
        config
            The `config` parameter is a dictionary that contains the configuration settings for the
        `MLPRegressor` class. It should have the following keys:
        '''
        
        super(MLPRegressor, self).__init__()
        self.depth = config["depth"]
        self.hidden_layer_widths = config["hidden_layer_widths"]
        self.fc_layers = nn.ModuleList()
        
        # Input layer
        self.fc_layers.append(nn.Linear(12, self.hidden_layer_widths[0]))
        
        # Hidden layers
        for i in range(self.depth - 1):
            self.fc_layers.append(nn.Linear(self.hidden_layer_widths[i], self.hidden_layer_widths[i+1]))
        
        # Output layer
        self.fc_layers.append(nn.Linear(self.hidden_layer_widths[-1], 1))
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        '''
        The forward function applies a ReLU activation function to the output of each fully connected
        layer, passing the input through the hidden layers and outputting the final prediction.
        
        Parameters
        ----------
        x
            The parameter `x` represents the input tensor to the forward method. It is passed through the
        fully connected layers with ReLU activation functions applied between them. The final output tensor
        represents the model's prediction.
        
        Returns
        -------
            the output tensor after passing it through the fully connected layers and applying the ReLU
        activation functions.
        '''
        
        for layer in self.fc_layers[:-1]:
            x = self.relu(layer(x))
        x = self.fc_layers[-1](x)
        
        return x.squeeze() # Squeeze the output tensor to remove the singleton dimension


class Trainer:
    
    '''
    The Trainer class is responsible for training a model, evaluating its performance, and recording
    metrics during training.
    '''

    def __init__(self, model, train_loader, val_loader, test_loader, config):
        
        '''
        The function initializes a class object with a model, data loaders, configuration settings,
        optimizer, loss function, and a writer for logging.
        
        Parameters
        ----------
        model
            The model parameter is the neural network model that you want to train. It should be an
        instance of a PyTorch model class, such as nn.Module or a custom model class that inherits from
        nn.Module. This model will be used to make predictions on the input data.
        train_loader
            The `train_loader` parameter is a DataLoader object that provides an iterable over the training
        dataset. It is used to load batches of training data during the training process.
        val_loader
            The `val_loader` parameter is a data loader object that is used to load the validation dataset.
        It is typically used during the training process to evaluate the performance of the model on a
        separate dataset and make decisions such as early stopping or hyperparameter tuning.
        test_loader
            The `test_loader` parameter is a data loader that is used to load the test dataset. It is
        typically used to evaluate the performance of the model on unseen data after training.
        config
            The `config` parameter is a dictionary that contains various configuration settings for the
        training process. It can include settings such as learning rate, batch size, number of epochs,
        etc. These settings can be used to customize the training process according to specific
        requirements.
        '''
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
        self.loss_function = nn.MSELoss()
        self.writer = SummaryWriter()

    def evaluate_model(self, data_loader):
        
        '''
        The `evaluate_model` function evaluates the performance of a machine learning model by
        calculating the root mean squared error (RMSE), R2 score, and inference latency.
        
        Parameters
        ----------
        data_loader
            The `data_loader` parameter is an instance of a PyTorch `DataLoader` class. It is used to load
        the data in batches during evaluation. The `data_loader` should be configured to load the
        evaluation dataset.
        
        Returns
        -------
            three values: rmse (root mean squared error), r2 (R-squared score), and inference_latency
        (inference latency).
        '''
        
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
        
        '''
        The `train` function trains a model for a specified number of epochs, calculates and saves
        various metrics, and returns a dictionary containing the metrics.
        
        Returns
        -------
            a dictionary called `metrics_dict` which contains the following metrics for the train, val, and
        test sets: RMSE_loss, R_squared, inference_latency.
        '''
        
        start_time = time.time()
        metrics_dict = {
            "train": {"RMSE_loss": None, "R_squared": None, "inference_latency": None},
            "val": {"RMSE_loss": None, "R_squared": None, "inference_latency": None},
            "test": {"RMSE_loss": None, "R_squared": None, "inference_latency": None}
        }

        for epoch in range(self.config["num_epochs"]):
            avg_train_loss = self.train_epoch(epoch)
            avg_val_loss = self.validate_epoch()
            avg_test_loss = self.test_epoch()

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

    def train_epoch(self, epoch):
        
        '''
        The function `train_epoch` trains a model for one epoch and returns the average training loss.
        
        Parameters
        ----------
        epoch
            The parameter "epoch" represents the current epoch number during training. An epoch is a
        complete pass through the entire training dataset.
        
        Returns
        -------
            the average training loss for the epoch.
        '''
        
        running_loss = 0.0

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
        
        return avg_train_loss

    def validate_epoch(self):
        
        '''
        The function calculates the average validation loss for a given epoch.
        
        Returns
        -------
            the average validation loss.
        '''
        
        with torch.no_grad():
            running_val_loss = 0.0

            for features, labels in self.val_loader:
                features = features.to(torch.float32)
                labels = labels.to(torch.float32)
                prediction = self.model(features)
                loss = torch.sqrt(self.loss_function(prediction, labels))
                running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(self.val_loader)
            
            return avg_val_loss

    def test_epoch(self):
        
        '''
        The function calculates the average test loss for a given model using a specified loss function
        and test data.
        
        Returns
        -------
            the average test loss.
        '''
        
        with torch.no_grad():
            running_test_loss = 0.0

            for features, labels in self.test_loader:
                features = features.to(torch.float32)
                labels = labels.to(torch.float32)
                prediction = self.model(features)
                loss = torch.sqrt(self.loss_function(prediction, labels))
                running_test_loss += loss.item()

            avg_test_loss = running_test_loss / len(self.test_loader)
            
            return avg_test_loss


def generate_nn_configs():
    
    '''
    The function `generate_nn_configs` generates a list of neural network configurations for a
    classification task, with different combinations of learning rates, hidden layer widths, number of
    epochs, and depths.
    
    Returns
    -------
        The function `generate_nn_configs` returns a list of neural network configurations.
    '''
    
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [4, 16, 32, 64]
    num_epochs = [100, 200, 500]
    depths = [2, 3, 4]

    nn_configs = []

    for config in itertools.product(learning_rates, hidden_layer_widths, num_epochs, depths):
        learning_rate, width, num_epoch, depth = config
        width_configurations = [width] * (depth - 1)
        nn_config = {
            "learning_rate": learning_rate,
            "hidden_layer_widths": [width] + width_configurations,
            "num_epochs": num_epoch,
            "depth": depth
        }
        nn_configs.append(nn_config)
        if len(nn_configs) >= 16:
            break

    return nn_configs


def find_best_nn(train_loader, val_loader, test_loader):
    
    '''
    The function `find_best_nn` trains multiple neural network models with different configurations and
    returns the best model, its metrics, and hyperparameters based on the validation loss.
    
    Parameters
    ----------
    train_loader
        A data loader for the training dataset. It is used to load batches of training data during the
    training process.
    val_loader
        The `val_loader` parameter is a data loader object that provides batches of validation data. It is
    used to evaluate the performance of the neural network model during training and select the best
    model based on the validation loss.
    test_loader
        The `test_loader` parameter is a data loader object that is used to load the test dataset. It is
    typically used to evaluate the performance of the trained model on unseen data.
    
    Returns
    -------
        the best trained model, the metrics dictionary for the best model, and the hyperparameters used for
    the best model.
    '''
    
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


if __name__ == "__main__":
    
    dataset = AirbnbNightlyPriceRegressionDataset()
    batch_size = 8

    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    train_dataset = AirbnbNightlyPriceRegressionDataset()
    train_dataset.X, train_dataset.y = X_train, y_train

    val_dataset = AirbnbNightlyPriceRegressionDataset()
    val_dataset.X, val_dataset.y = X_val, y_val

    test_dataset = AirbnbNightlyPriceRegressionDataset()
    test_dataset.X, test_dataset.y = X_test, y_test

    config = get_nn_config()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    
    # Train the model and save metrics
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

