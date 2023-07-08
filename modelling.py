import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from tabular_data import load_airbnb
import itertools
import os
import joblib
import json


# Load the tabular data using load_airbnb function
features, labels = load_airbnb(label="Price_Night")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

y_train = y_train.values.ravel()  # Convert y_train to 1-dimensional array
y_valid = y_valid.values.ravel()  # Convert y_valid to 1-dimensional array
y_test = y_test.values.ravel()  # Convert y_test to 1-dimensional array


def train_linear_regression():
    
    metrics = {}
    
    # Initialize and train the linear regression model
    model = SGDRegressor()
    model.fit(X_train, y_train)
    hyperparameters = model.get_params()

    # Compute predictions for the training set
    train_predictions = model.predict(X_train)

    # Compute predictions for the test set
    test_predictions = model.predict(X_test)

    # Compute RMSE for the training set
    metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, train_predictions))

    # Compute RMSE for the test set
    metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, test_predictions))

    # Compute R^2 score for the training set
    metrics["train_R2"] = r2_score(y_train, train_predictions)

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
    best_metrics = {"validation_RMSE": float('inf')}
    
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
        if rmse < best_metrics["validation_RMSE"]:
            best_model = model
            best_params = params
            best_metrics["validation_RMSE"] = rmse
    
    # Calculate additional performance metrics for the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    best_metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    best_metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    best_metrics["train_R2"] = best_model.score(X_train, y_train)
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


def tune_regression_model_hyperparameters():
    
    best_metrics = {}
    
    model = SGDRegressor()
    
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
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Calculate additional performance metrics for the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    best_metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    best_metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    best_metrics["train_R2"] = best_model.score(X_train, y_train)
    best_metrics["test_R2"] = best_model.score(X_test, y_test)
    
    print("--------------------------------")
    print("SKLEARN GRID SEARCH")
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


def save_model(model, hyperparameters, metrics, folder, model_filename, params_filename, metrics_filename):
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the model
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



def main(models_to_run):

    if 'baseline' in models_to_run:
        
        baseline_model, baseline_params, baseline_metrics = train_linear_regression()
        
        save_model(baseline_model, baseline_params, baseline_metrics, 'models/regression/linear_regression/baseline_sgd/',
                   'baseline_sgd_model', 'baseline_sgd_params', 'baseline_sgd_metrics')

    if 'custom' in models_to_run:
        
        custom_model_class = SGDRegressor
        
        custom_hyperparameters = {
        'loss': ['squared_error', 'huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'max_iter': [1000, 2000],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2],
        'tol': [1e-3, 1e-4]}
        
        custom_model, custom_params, custom_metrics = custom_tune_regression_model_hyperparameters(
            custom_model_class, custom_hyperparameters)
        
        save_model(custom_model, custom_params, custom_metrics, 'models/regression/linear_regression/custom_sgd/',
                   'custom_sgd_model', 'custom_sgd_params', 'custom_sgd_metrics')
    
    if 'sklearn' in models_to_run:
        
        sklearn_model, sklearn_params, sklearn_metrics = tune_regression_model_hyperparameters()
        
        save_model(sklearn_model, sklearn_params, sklearn_metrics, 'models/regression/linear_regression/sklearn_sgd/',
                    'sklearn_sgd_model', 'sklearn_sgd_params', 'sklearn_sgd_metrics')


if __name__ == "__main__":
    main(['baseline', 'custom', 'sklearn'])