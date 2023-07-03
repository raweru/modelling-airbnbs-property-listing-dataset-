import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from tabular_data import load_airbnb
import itertools
import os
import joblib
import json


def train_linear_regression():
    # Load the tabular data using load_airbnb function
    features, labels = load_airbnb(label="Price_Night")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=4)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    y_train = y_train.values.ravel()  # Convert y_train to 1-dimensional array
    y_valid = y_valid.values.ravel()  # Convert y_valid to 1-dimensional array
    y_test = y_test.values.ravel()  # Convert y_test to 1-dimensional array
    
    # Initialize and train the linear regression model
    model = SGDRegressor()
    model.fit(X_train, y_train)

    # Compute predictions for the training set
    train_predictions = model.predict(X_train)

    # Compute predictions for the test set
    test_predictions = model.predict(X_test)

    # Compute RMSE for the training set
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

    # Compute RMSE for the test set
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    # Compute R^2 score for the training set
    train_r2 = r2_score(y_train, train_predictions)

    # Compute R^2 score for the test set
    test_r2 = r2_score(y_test, test_predictions)

    # Print the performance measures
    print("--------------------------------")
    print("BASELINE MODEL (NO FINE TUNING)")
    print("Training set RMSE: ", train_rmse)
    print("Test set RMSE: ", test_rmse)
    print("Training set R^2: ", train_r2)
    print("Test set R^2: ", test_r2)
    print("--------------------------------")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def generate_parameter_combinations(hyperparameters):
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_valid, y_valid, X_test, y_test, hyperparameters):
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
    print(f"Best model:{best_model}")
    print(f"Best params:{best_params}")
    print(f"Best metrics:{best_metrics}")
    print("--------------------------------")
    
    return best_model, best_params, best_metrics


def tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test):
    
    best_metrics = {}
    
    model = SGDRegressor()
    
    param_grid = {
        'loss': ['squared_loss', 'huber'],
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
    print("SKLEARN GRIDSEARCHCV")
    print("Best Hyperparameters:")
    print(best_params)
    print(f"Best Metrics: {best_metrics}")
    print("--------------------------------")
    
    return best_model, best_params, best_metrics


def save_model(model, folder):
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the model
    model_path = os.path.join(folder, "model.joblib")
    joblib.dump(model, model_path)

    # Extract hyperparameters from the model
    hyperparameters = model.get_params()

    # Save the hyperparameters
    hyperparameters_path = os.path.join(folder, "hyperparameters.json")
    with open(hyperparameters_path, "w") as file:
        json.dump(hyperparameters, file)

    # Calculate and extract metrics from the model
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "R2": r2
    }

    # Save the metrics
    metrics_path = os.path.join(folder, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)



if __name__ == "__main__":
    
    #TODO: Train baseline model
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_linear_regression()

    """Training set RMSE:  1158344774.6662142
    Test set RMSE:  1228675273.35997
    Training set R^2:  -74626863578953.1
    Test set R^2:  -111151183104622.05"""

    #TODO: Custom fine tune grid search
    # Define the hyperparameters to be tuned
    hyperparameters = {
        'loss': ['squared_loss', 'huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'max_iter': [1000, 2000],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2],
        'tol': [1e-3, 1e-4]
    }

    # Create an instance of the model class (e.g., SGDRegressor)
    model_class = SGDRegressor

    # Call the custom grid search function
    best_model, best_params, best_metrics = custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_valid, y_valid, X_test, y_test, hyperparameters)

    """ Best model:SGDRegressor(alpha=0.1, learning_rate='adaptive')
        Best params:{'alpha': 0.1, 'learning_rate': 'adaptive'}
        Best metrics:{'validation_RMSE': 143459528.71772546, 'train_RMSE': 164706481.42072955, 
        'test_RMSE': 146281798.80221358, 'train_R2': -1508831981312.8716, 'test_R2': -1295221330677.812}"""

    #TODO: sklearn fine tune grid search
    
    tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test)