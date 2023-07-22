import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from tabular_data import load_airbnb
import itertools
import os
import joblib
import json
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
import torch

# Ignore the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ignore the data conversion warning
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Load the tabular data using load_airbnb function
features, labels = load_airbnb(label="Price_Night")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


def train_linear_regression():
    
    metrics = {}
    
    # Initialize and train the linear regression model
    model = SGDRegressor()
    model.fit(X_train, y_train)
    hyperparameters = model.get_params()

    # Compute predictions for the training set
    train_predictions = model.predict(X_train)
    
    # Compute predictions for the validation set
    valid_predictions = model.predict(X_valid)

    # Compute predictions for the test set
    test_predictions = model.predict(X_test)
    
    # Compute RMSE for the training set
    metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, train_predictions))
    
    # Compute RMSE for the validation set
    metrics["valid_RMSE"] = np.sqrt(mean_squared_error(y_valid, valid_predictions))

    # Compute RMSE for the test set
    metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, test_predictions))

    # Compute R^2 score for the training set
    metrics["train_R2"] = r2_score(y_train, train_predictions)
    
    # Compute R^2 score for the validation set
    metrics["valid_R2"] = r2_score(y_valid, valid_predictions)

    # Compute R^2 score for the test set
    metrics["test_R2"] = r2_score(y_test, test_predictions)

    # Print the performance measures
    print("--------------------------------")
    print("BASELINE MODEL, NO FINE TUNING")
    print(f"Model:{model}")
    print("")
    print("Hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"{param}: {value}")
    print("")
    print("Metrics:")
    for param, value in metrics.items():
        print(f"{param}: {value}")
    print("--------------------------------")
    
    return model, hyperparameters, metrics


def generate_parameter_combinations(hyperparameters):
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def custom_tune_regression_model_hyperparameters(model_class, hyperparameters):
    best_model = None
    best_params = {}
    best_metrics = {"valid_RMSE": float('inf')}
    
    # Iterate over hyperparameter combinations
    for params in generate_parameter_combinations(hyperparameters):
        # Create an instance of the model with the current hyperparameters
        model = model_class(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the validation set
        y_pred = model.predict(X_valid)
        
        # Calculate RMSE on the validation set
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        
        # Check if this model is the best so far
        if rmse < best_metrics["valid_RMSE"]:
            best_model = model
            best_params = params
            best_metrics["valid_RMSE"] = rmse
    
    # Calculate additional performance metrics for the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    best_metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    best_metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    best_metrics["train_R2"] = best_model.score(X_train, y_train)
    best_metrics["valid_R2"] = best_model.score(X_valid, y_valid)
    best_metrics["test_R2"] = best_model.score(X_test, y_test)
    
    
    print("--------------------------------")
    print("CUSTOM GRID SEARCH")
    print(f"Best Model:{best_model}")
    print("")
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("")
    print("Best Metrics:")
    for param, value in best_metrics.items():
        print(f"{param}: {value}")
    print("--------------------------------")
    
    return best_model, best_params, best_metrics


def tune_regression_model_hyperparameters(model_class, param_grid):
    
    best_metrics = {}

    model = model_class()
    

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Calculate additional performance metrics for the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_valid = best_model.predict(X_valid)
    y_pred_test = best_model.predict(X_test)
    
    best_metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    best_metrics["valid_RMSE"] = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    best_metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    best_metrics["train_R2"] = best_model.score(X_train, y_train)
    best_metrics["valid_R2"] = best_model.score(X_valid, y_valid)
    best_metrics["test_R2"] = best_model.score(X_test, y_test)
    
    print("--------------------------------")
    print(f"SKLEARN GRID SEARCH - {model_class}")
    print(f"Best Model:{best_model}")
    print("")
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("")
    print("Best Metrics:")
    for param, value in best_metrics.items():
        print(f"{param}: {value}")
    print("--------------------------------")
    if model == RandomForestRegressor():
        feature_importances = model.feature_importances_

        # Sort the feature importances in descending order
        indices = np.argsort(feature_importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(X_train.shape[1]):
            print(f"{f+1}. Feature '{X_train.columns[indices[f]]}': {feature_importances[indices[f]]}")


    
    return best_model, best_params, best_metrics


def save_model(model, hyperparameters, metrics, folder, model_filename=None, params_filename=None, metrics_filename=None):
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    if isinstance(model, torch.nn.Module):  # Check if the model is a PyTorch model
        # Save the model
        model_path = os.path.join(folder, "model.pt")
        torch.save(model.state_dict(), model_path)

        # Save the hyperparameters
        hyperparameters_path = os.path.join(folder, "hyperparameters.json")
        with open(hyperparameters_path, "w") as file:
            json.dump(hyperparameters, file)

        # Save the metrics
        metrics_path = os.path.join(folder, "metrics.json")
        with open(metrics_path, "w") as file:
            json.dump(metrics, file)
    else:
        # Save the non-PyTorch model using joblib
        model_path = os.path.join(folder, f"{model_filename}.joblib")
        joblib.dump(model, model_path)

        # Save the hyperparameters
        hyperparameters_path = os.path.join(folder, f"{params_filename}.json")
        with open(hyperparameters_path, "w") as file:
            json.dump(hyperparameters, file)

        # Save the metrics
        metrics_path = os.path.join(folder, f"{metrics_filename}.json")
        with open(metrics_path, "w") as file:
            json.dump(metrics, file)


def evaluate_all_models(models_to_run, task_folder):

    if 'baseline_sgd' in models_to_run:
        
        baseline_model, baseline_params, baseline_metrics = train_linear_regression()
        
        save_model(baseline_model, baseline_params, baseline_metrics, f'{task_folder}/baseline_sgd/',
                   'baseline_sgd_model', 'baseline_sgd_params', 'baseline_sgd_metrics')

    if 'custom_ft_sgd' in models_to_run:
        
        custom_model_class = SGDRegressor
        
        param_grid = {
        'loss': ['squared_error', 'huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'max_iter': [1000, 2000],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2],
        'tol': [1e-3, 1e-4]}
        
        custom_model, custom_params, custom_metrics = custom_tune_regression_model_hyperparameters(
            custom_model_class, param_grid)
        
        save_model(custom_model, custom_params, custom_metrics, f'{task_folder}/custom_ft_sgd/',
                   'custom_ft_sgd_model', 'custom_ft_sgd_params', 'custom_ft_sgd_metrics')
    
    if 'sklearn_ft_sgd' in models_to_run:
        
        model_class = SGDRegressor
        param_grid = {
            'loss': ['squared_error', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'max_iter': [1000, 2000],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2],
            'tol': [1e-3, 1e-4]
        }
        
        sklearn_model, sklearn_params, sklearn_metrics = tune_regression_model_hyperparameters(
            model_class, param_grid)
        
        save_model(sklearn_model, sklearn_params, sklearn_metrics, f'{task_folder}/sklearn_ft_sgd/',
                    'sklearn_ft_sgd_model', 'sklearn_ft_sgd_params', 'sklearn_ft_sgd_metrics')

    if 'decision_tree' in models_to_run:
        
        model_class = DecisionTreeRegressor
        param_grid = {
            'max_depth': [None, 5, 10],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }
        
        sklearn_model, sklearn_params, sklearn_metrics = tune_regression_model_hyperparameters(
            model_class, param_grid)
        
        save_model(sklearn_model, sklearn_params, sklearn_metrics, f'{task_folder}/decision_tree/',
                    'decision_tree_model', 'decision_tree_params', 'decision_tree_metrics')

    if 'random_forest' in models_to_run:
        
        model_class = RandomForestRegressor
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        
        sklearn_model, sklearn_params, sklearn_metrics = tune_regression_model_hyperparameters(
            model_class, param_grid)
        
        save_model(sklearn_model, sklearn_params, sklearn_metrics, f'{task_folder}/random_forest/',
                    'random_forest_model', 'random_forest_params', 'random_forest_metrics')

    if 'gboost' in models_to_run:
        
        model_class = GradientBoostingRegressor
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 7, 15],
            'min_samples_leaf': [1, 2, 4]
        }
        
        sklearn_model, sklearn_params, sklearn_metrics = tune_regression_model_hyperparameters(
            model_class, param_grid)
        
        save_model(sklearn_model, sklearn_params, sklearn_metrics, f'{task_folder}/gboost/',
                    'gboost_model', 'gboost_params', 'gboost_metrics')


def find_best_model(models_to_evaluate):
    
    best_model = None
    best_params = {}
    best_metrics = {}

    for model_name in models_to_evaluate:
        model_folder = f"models/regression/{model_name}/"

        # Load the model, hyperparameters, and metrics
        model_path = os.path.join(model_folder, f"{model_name}_model.joblib")
        model = joblib.load(model_path)

        hyperparameters_path = os.path.join(model_folder, f"{model_name}_params.json")
        with open(hyperparameters_path, "r") as file:
            hyperparameters = json.load(file)

        metrics_path = os.path.join(model_folder, f"{model_name}_metrics.json")
        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        # Update the best model based on the performance metrics
        if best_model is None or metrics["valid_RMSE"] < best_metrics["valid_RMSE"]:
            best_model = model
            best_params = hyperparameters
            best_metrics = metrics
            
    print(best_model, best_params, best_metrics)
    
    
    return best_model, best_params, best_metrics


if __name__ == "__main__":
    
    models = [
        "baseline_sgd", "custom_ft_sgd", "sklearn_ft_sgd", "decision_tree", "random_forest", "gboost"]
    
    evaluate_all_models(models, 'models/regression')
    
    find_best_model(models)