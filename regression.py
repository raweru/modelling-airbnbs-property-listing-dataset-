import itertools
import joblib
import json
import numpy as np
import os
import torch
import torch.nn as nn
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from tabular_data import load_airbnb


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


def load_and_split_data(label="Price_Night", test_size=0.3, valid_size=0.5, random_state=42):
    
    '''
    The function `load_and_split_data` loads tabular data using the `load_airbnb` function and splits it
    into training, validation, and testing sets.
    
    Parameters
    ----------
    label, optional
        The label parameter specifies the target variable or the variable that we want to predict. In this
    case, it is set to "Price_Night", which means we want to predict the price per night for Airbnb
    listings.
    test_size
        The test_size parameter determines the proportion of the data that will be used for testing. It is
    a float value between 0 and 1, where 0.3 means that 30% of the data will be used for testing.
    valid_size
        The `valid_size` parameter represents the proportion of the data that should be allocated for the
    validation set. For example, if `valid_size=0.5`, it means that 50% of the data will be used for
    validation.
    random_state, optional
        The random_state parameter is used to set the seed for the random number generator. This ensures
    that the data is split in the same way every time the function is called, which is useful for
    reproducibility.
    
    Returns
    -------
        The function load_and_split_data returns six variables: X_train, y_train, X_valid, y_valid, X_test,
    and y_test. These variables represent the training, validation, and testing sets of the tabular
    data.
    '''
    
    features, labels = load_airbnb(label=label)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=valid_size, random_state=random_state)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_linear_regression():
    
    '''
    The function `train_linear_regression` trains a linear regression model using stochastic gradient
    descent and computes various performance metrics for the training, validation, and test sets.
    
    Returns
    -------
        the trained linear regression model, the hyperparameters of the model, and the performance metrics
    (RMSE and R^2 scores) for the training, validation, and test sets.
    '''
    
    metrics = {}
    
    model = SGDRegressor()
    model.fit(X_train, y_train)
    hyperparameters = model.get_params()

    train_predictions = model.predict(X_train)
    
    valid_predictions = model.predict(X_valid)

    test_predictions = model.predict(X_test)
    
    metrics["train_RMSE"] = np.sqrt(mean_squared_error(y_train, train_predictions))
    
    metrics["valid_RMSE"] = np.sqrt(mean_squared_error(y_valid, valid_predictions))

    metrics["test_RMSE"] = np.sqrt(mean_squared_error(y_test, test_predictions))

    metrics["train_R2"] = r2_score(y_train, train_predictions)
    
    metrics["valid_R2"] = r2_score(y_valid, valid_predictions)

    metrics["test_R2"] = r2_score(y_test, test_predictions)

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
    
    '''
    The function generates all possible combinations of values for a given set of hyperparameters.
    
    Parameters
    ----------
    hyperparameters
        A dictionary containing the hyperparameters for a model. The keys are the names of the
    hyperparameters, and the values are lists of possible values for each hyperparameter.
    '''
    
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def custom_tune_regression_model_hyperparameters(model_class, hyperparameters):
    
    '''
    The function `custom_tune_regression_model_hyperparameters` performs a custom grid search to find
    the best hyperparameters for a regression model and returns the best model, best hyperparameters,
    and performance metrics.
    
    Parameters
    ----------
    model_class
        The `model_class` parameter is the class of the regression model that you want to tune. It should
    be a class that implements the necessary methods for training and making predictions, such as
    `fit()` and `predict()`. Examples of regression model classes include `LinearRegression`,
    `RandomForestRegressor`,
    hyperparameters
        The `hyperparameters` parameter is a dictionary that contains the hyperparameters for the
    regression model. Each key in the dictionary represents a hyperparameter name, and the corresponding
    value is a list of possible values for that hyperparameter. The function
    `generate_parameter_combinations` is used to generate all possible combinations
    
    Returns
    -------
        the best model, the best hyperparameters, and the best metrics.
    '''
    
    best_model = None
    best_params = {}
    best_metrics = {"valid_RMSE": float('inf')}
    
    # Iterate over hyperparameter combinations
    for params in generate_parameter_combinations(hyperparameters):
        # Create an instance of the model with the current hyperparameters
        model = model_class(**params)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_valid)
        
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
    
    '''
    The function `tune_regression_model_hyperparameters` performs grid search to find the best
    hyperparameters for a given regression model class and returns the best model, best hyperparameters,
    and performance metrics.
    
    Parameters
    ----------
    model_class
        The model_class parameter is the class of the regression model that you want to tune. It should be
    a class object, such as RandomForestRegressor or LinearRegression.
    param_grid
        The `param_grid` parameter is a dictionary that specifies the hyperparameters and their
    corresponding values to be tuned for the regression model. It is used in the `GridSearchCV` function
    to perform a grid search over the specified hyperparameters.
    
    Returns
    -------
        the best model found during the grid search, the best hyperparameters for that model, and the
    performance metrics (RMSE and R2) for the best model on the training, validation, and test sets.
    '''
    
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
    
    '''
    The `save_model` function saves a machine learning model, its hyperparameters, and metrics to a
    specified folder.
    
    Parameters
    ----------
    model
        The `model` parameter is the machine learning model that you want to save. It can be either a
    PyTorch model (subclass of `torch.nn.Module`) or a non-PyTorch model.
    hyperparameters
        The `hyperparameters` parameter is a dictionary that contains the hyperparameters used to train the
    model. It typically includes values such as learning rate, batch size, number of epochs, etc.
    metrics
        The "metrics" parameter is a dictionary that contains the evaluation metrics of the model. It
    typically includes metrics such as accuracy, precision, recall, F1 score, etc. These metrics are
    used to evaluate the performance of the model on the validation or test set.
    folder
        The `folder` parameter is the directory where you want to save the model, hyperparameters, and
    metrics.
    model_filename
        The `model_filename` parameter is the name of the file where the model will be saved. If this
    parameter is not provided, the model will be saved with the default name "model.pt" for PyTorch
    models or with the default name "{model_filename}.joblib" for non-Py
    params_filename
        The `params_filename` parameter is the name of the file where the hyperparameters will be saved. It
    is used to specify the filename for the JSON file that will store the hyperparameters of the model.
    metrics_filename
        The `metrics_filename` parameter is the name of the file where the metrics will be saved. It is a
    string that should not include the file extension. For example, if you want to save the metrics in a
    file named "evaluation_metrics.json", you would pass "evaluation_metrics" as the value
    '''
    
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
    '''
    The function `evaluate_all_models` takes a list of models to run and a task folder as input, and for
    each model in the list, it tunes the hyperparameters, saves the model, hyperparameters, and metrics
    in the task folder.
    
    Parameters
    ----------
    models_to_run : list
        A list of models to run. Available options: 'baseline_sgd', 'custom_ft_sgd', 'sklearn_ft_sgd',
        'decision_tree', 'random_forest', 'gboost'.
    task_folder : str
        The folder where the models and their corresponding parameters and metrics will be saved.
    '''

    model_info = {
        'baseline_sgd': (train_linear_regression, 'baseline_sgd_model', 'baseline_sgd_params', 'baseline_sgd_metrics'),
        'custom_ft_sgd': (custom_tune_regression_model_hyperparameters, SGDRegressor, {
            'loss': ['squared_error', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'max_iter': [1000, 2000],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2],
            'tol': [1e-3, 1e-4]
        }, 'custom_ft_sgd_model', 'custom_ft_sgd_params', 'custom_ft_sgd_metrics'),
        'sklearn_ft_sgd': (SGDRegressor, {
            'loss': ['squared_error', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'max_iter': [1000, 2000],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2],
            'tol': [1e-3, 1e-4]
        }, 'sklearn_ft_sgd_model', 'sklearn_ft_sgd_params', 'sklearn_ft_sgd_metrics'),
        'decision_tree': (DecisionTreeRegressor, {
            'max_depth': [None, 5, 10],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }, 'decision_tree_model', 'decision_tree_params', 'decision_tree_metrics'),
        'random_forest': (RandomForestRegressor, {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }, 'random_forest_model', 'random_forest_params', 'random_forest_metrics'),
        'gboost': (GradientBoostingRegressor, {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 7, 15],
            'min_samples_leaf': [1, 2, 4]
        }, 'gboost_model', 'gboost_params', 'gboost_metrics')
    }

    for model_name in models_to_run:
        model_info_entry = model_info[model_name]
        if isinstance(model_info_entry[0], type):
            custom_model_class = model_info_entry[0]
            param_grid = model_info_entry[1]
            model_class = custom_tune_regression_model_hyperparameters
        else:
            model_class = model_info_entry[0]
            param_grid = model_info_entry[1]
        
        model, params, metrics = tune_regression_model_hyperparameters(
            model_class, param_grid)
        
        save_model(model, params, metrics, f'{task_folder}/{model_name}/',
                   model_info_entry[2], model_info_entry[3], model_info_entry[4])


def find_best_model(models_to_evaluate):
    
    '''
    The function `find_best_model` takes a list of model names, loads the models, hyperparameters, and
    metrics from the corresponding files, and returns the best model, its hyperparameters, and its
    metrics based on the validation RMSE.
    
    Parameters
    ----------
    models_to_evaluate
        A list of model names to evaluate. Each model name corresponds to a folder in the
    "models/regression" directory, where the model, hyperparameters, and metrics are stored.
    
    Returns
    -------
        The function `find_best_model` returns three values: `best_model`, `best_params`, and
    `best_metrics`.
    '''
    
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
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_split_data(label="Price_Night")
    
    evaluate_all_models(models, 'models/regression')
    
    find_best_model(models)