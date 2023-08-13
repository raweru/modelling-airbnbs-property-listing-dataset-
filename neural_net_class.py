import itertools
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tabular_data import load_airbnb


class AirbnbListingClassifierDataset(Dataset):
    
    '''
    The `AirbnbListingClassifierDataset` class is a PyTorch dataset that loads Airbnb listing data and
    returns features and labels for classification.
    '''
    
    def __init__(self):
        
        '''
        The function initializes the X and y variables by loading the Airbnb dataset and converting them
        into torch tensors.
        '''
        
        super().__init__()
        self.X, self.y = load_airbnb(label="bedrooms")
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __getitem__(self, idx):
        
        '''
        The `__getitem__` function returns the `X` and `y` values at the specified index.
        
        Parameters
        ----------
        idx
            The parameter "idx" is the index of the item that you want to retrieve from the object. It is
        used to access specific elements of the "X" and "y" attributes of the object.
        
        Returns
        -------
            The code is returning the X and y values at the given index.
        '''
        
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        
        '''
        The above function returns the length of the attribute "X" of the object.
        
        Returns
        -------
            The length of the attribute `self.X` is being returned.
        '''
        
        return len(self.X)


class MLPClassifier(nn.Module):
    
    '''
    The MLPClassifier class is a multi-layer perceptron classifier that takes in a configuration
    dictionary and performs forward propagation on input data.
    '''
    
    def __init__(self, config):
        
        '''
        The above function initializes a multi-layer perceptron classifier with the specified
        configuration.
        
        Parameters
        ----------
        config
            The `config` parameter is a dictionary that contains the following keys:
        '''
        
        super(MLPClassifier, self).__init__()
        self.depth = config["depth"]
        self.hidden_layer_widths = config["hidden_layer_widths"]
        self.num_classes = config["num_classes"]
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(12, self.hidden_layer_widths[0]))

        for i in range(self.depth - 1):
            self.fc_layers.append(nn.Linear(self.hidden_layer_widths[i], self.hidden_layer_widths[i + 1]))

        self.fc_layers.append(nn.Linear(self.hidden_layer_widths[-1], self.num_classes))
        self.relu = nn.ReLU()

    def forward(self, x):
        
        '''
        The forward function applies a series of fully connected layers with ReLU activation to the
        input tensor x and returns the final output.
        
        Parameters
        ----------
        x
            The parameter `x` represents the input to the forward method. It is the input data that will be
        passed through the layers of the neural network.
        
        Returns
        -------
            The output of the last fully connected layer.
        '''
        
        for layer in self.fc_layers[:-1]:
            x = self.relu(layer(x))
        x = self.fc_layers[-1](x)
        
        return x


class Trainer:
    
    '''
    The Trainer class is responsible for training and evaluating a model using a given train,
    validation, and test dataset.
    '''
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        
        '''
        The function initializes a class object with a model, data loaders, configuration parameters,
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
        self.loss_function = nn.CrossEntropyLoss()
        self.writer = SummaryWriter()

    def evaluate_model(self, data_loader):
        
        '''
        The function evaluates a machine learning model using a given data loader and returns the
        accuracy, average loss, and inference latency.
        
        Parameters
        ----------
        data_loader
            The `data_loader` parameter is an instance of a PyTorch `DataLoader` object. It is used to load
        the data in batches during evaluation. The `DataLoader` object should be configured with the
        appropriate dataset and batch size.
        
        Returns
        -------
            three values: accuracy, average loss, and inference latency.
        '''
        
        n_samples = len(data_loader.dataset)
        start_time = time.time()
        self.model.eval()
        correct_predictions = 0
        total_examples = 0
        total_loss = 0.0

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(torch.float32)
                targets = targets.to(torch.long)

                output = self.model(features)
                _, predictions = torch.max(output, dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_examples += targets.size(0)

                loss = self.loss_function(output, targets)
                total_loss += loss.item()

        end_time = time.time()
        inference_latency = (end_time - start_time) / n_samples
        accuracy = correct_predictions / total_examples
        avg_loss = total_loss / len(data_loader)
        
        return accuracy, avg_loss, inference_latency


    def train(self):
        
        '''
        The `train` function trains a model for a specified number of epochs, evaluates its performance
        on the validation and test sets, and returns a dictionary of metrics.
        
        Returns
        -------
            a dictionary called `metrics_dict` which contains the following metrics for the last epoch:
        '''
        
        start_time = time.time()
        metrics_dict = {
            "train": {"accuracy": None, "inference_latency": None, "cross_entropy_loss": None},
            "val": {"accuracy": None, "inference_latency": None, "cross_entropy_loss": None},
            "test": {"accuracy": None, "inference_latency": None, "cross_entropy_loss": None}
        }

        for epoch in range(self.config["num_epochs"]):
            avg_train_loss = self.train_epoch(epoch)
            avg_val_accuracy, avg_val_loss, val_inference_latency = self.evaluate_model(self.val_loader)
            avg_test_accuracy, avg_test_loss, test_inference_latency = self.evaluate_model(self.test_loader)

            # Print epoch information
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {avg_val_accuracy:.4f} | Test Accuracy: {avg_test_accuracy:.4f}")

            # Save the metrics for the last epoch only
            if epoch == self.config['num_epochs'] - 1:
                metrics_dict["train"]["accuracy"] = avg_val_accuracy
                metrics_dict["train"]["inference_latency"] = val_inference_latency
                metrics_dict["train"]["cross_entropy_loss"] = avg_train_loss
                metrics_dict["val"]["accuracy"] = avg_val_accuracy
                metrics_dict["val"]["inference_latency"] = val_inference_latency
                metrics_dict["val"]["cross_entropy_loss"] = avg_val_loss
                metrics_dict["test"]["accuracy"] = avg_test_accuracy
                metrics_dict["test"]["inference_latency"] = test_inference_latency
                metrics_dict["test"]["cross_entropy_loss"] = avg_test_loss

            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Accuracy/Val', avg_val_accuracy, epoch)

        end_time = time.time()
        training_duration = end_time - start_time
        self.training_duration = training_duration
        self.writer.close()

        return metrics_dict


    def train_epoch(self, epoch):
        
        '''
        The function `train_epoch` trains a model for one epoch by iterating over the training data,
        computing predictions, calculating the loss, and updating the model's parameters.
        
        Parameters
        ----------
        epoch
            The parameter "epoch" represents the current epoch number during training. An epoch is a
        complete pass through the entire training dataset.
        
        Returns
        -------
            The average training loss for the epoch.
        '''
        
        running_loss = 0.0

        self.model.train()
        for features, labels in self.train_loader:
            features = features.to(torch.float32)
            labels = labels.to(torch.long)
            prediction = self.model(features)
            loss = self.loss_function(prediction, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(self.train_loader)
        
        return avg_train_loss

    def validate_epoch(self):
        
        '''
        The function "validate_epoch" calculates the accuracy of a model using a validation loader.
        
        Returns
        -------
            The accuracy of the model on the validation data is being returned.
        '''
        
        accuracy = self.evaluate_model(self.val_loader)
        
        return accuracy


def generate_nn_configs():
    
    '''
    The function `generate_nn_configs` generates a list of neural network configurations based on
    different combinations of learning rates, hidden layer widths, number of epochs, and depths.
    
    Returns
    -------
        The function `generate_nn_configs` returns a list of neural network configurations.
    '''
    
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [25, 50, 100]
    num_epochs = [100, 200, 500]
    depths = [2, 3, 4]
    num_classes = 12

    configs = list(itertools.product(learning_rates, num_epochs, depths))

    nn_configs = []
    for config in configs:
        for width_config in itertools.product(hidden_layer_widths, repeat=config[2] - 1):
            nn_config = {
                "learning_rate": config[0],
                "hidden_layer_widths": [config[1]] + list(width_config),
                "num_epochs": config[1],
                "depth": config[2],
                "num_classes": num_classes
            }
            nn_configs.append(nn_config)
            if len(nn_configs) >= 16:
                break
        if len(nn_configs) >= 16:
            break

    return nn_configs


def find_best_nn(train_loader, val_loader, test_loader):
    
    '''
    The function `find_best_nn` trains multiple neural network models with different configurations and
    returns the best performing model, its evaluation metrics, and the hyperparameters used.
    
    Parameters
    ----------
    train_loader
        The train_loader parameter is a data loader object that provides batches of training data to the
    model during training. It is used to iterate over the training dataset in mini-batches.
    val_loader
        The `val_loader` parameter is a data loader object that is used to load the validation dataset. It
    is typically used during the training process to evaluate the performance of the model on a separate
    dataset and make decisions such as early stopping or selecting the best model.
    test_loader
        The `test_loader` parameter is a data loader object that is used to load the test dataset. It is
    typically used to evaluate the performance of the trained model on unseen data.
    
    Returns
    -------
        the best model, best metrics, and best hyperparameters.
    '''
    
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_val_accuracy = 0.0

    configs = generate_nn_configs()
    for i, config in enumerate(configs):
        print(f"Training model {i + 1} out of {len(configs)}...")
        model = MLPClassifier(config)
        trainer = Trainer(model, train_loader, val_loader, test_loader, config)
        metrics_dict = trainer.train()

        val_accuracy = metrics_dict["val"]["accuracy"]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_metrics = metrics_dict
            best_hyperparameters = config

        # Stop after training 16 models
        if i == 15:
            break

    return best_model, best_metrics, best_hyperparameters


if __name__ == "__main__":
    dataset = AirbnbListingClassifierDataset()
    batch_size = 8
    num_classes = 12  # Number of classes in your classification task

    # Split the dataset into training, validation, and test sets
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Create the training dataset
    train_dataset = AirbnbListingClassifierDataset()
    train_dataset.X, train_dataset.y = X_train, y_train

    # Create the validation dataset
    val_dataset = AirbnbListingClassifierDataset()
    val_dataset.X, val_dataset.y = X_val, y_val

    # Create the test dataset
    test_dataset = AirbnbListingClassifierDataset()
    test_dataset.X, test_dataset.y = X_test, y_test

    # Create the data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # Train the model and save metrics
    best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader, val_loader, test_loader)

    # Save the best model, its metrics, and hyperparameters
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("models", "neural_networks", "classification", current_time)
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
