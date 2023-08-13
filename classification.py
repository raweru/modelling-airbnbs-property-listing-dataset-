import json
import joblib
import os
import warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from regression import load_and_split_data, save_model


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


def train_logistic_regression():
    
    '''
    The function `train_logistic_regression` trains a logistic regression model, computes predictions
    for the training, validation, and test sets, and prints the classification report and accuracy for
    each set.
    '''
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    
    valid_predictions = model.predict(X_valid)

    test_predictions = model.predict(X_test)
    
    print("--------------------------------")
    print("--------------------------------")
    print("BASELINE LOGISTIC REGRESSION")
    print("--------------------------------")
    print("TRAINING SET")
    print(classification_report(y_train, train_predictions)) 
    print(f"Accuracy: {accuracy_score(y_train, train_predictions)}")
    print("--------------------------------")
    print("VALIDATION SET")
    print(classification_report(y_valid, valid_predictions))
    print(f"Accuracy: {accuracy_score(y_valid, valid_predictions)}")
    print("--------------------------------")
    print("TEST SET")
    print(classification_report(y_test, test_predictions))
    print(f"Accuracy: {accuracy_score(y_test, test_predictions)}")
    print("--------------------------------")
    print("--------------------------------")


def tune_classification_model_hyperparameters(model_class, param_grid):
    
    '''
    The function tunes the hyperparameters of a classification model using grid search and returns the
    best model, best hyperparameters, and performance metrics.
    
    Parameters
    ----------
    model_class
        The `model_class` parameter is the class of the classification model that you want to tune the
    hyperparameters for. It should be a class object, such as `RandomForestClassifier` or
    `LogisticRegression`.
    param_grid
        The `param_grid` is a dictionary that contains the hyperparameters and their corresponding values
    that you want to tune for the classification model. It should be in the following format:
    
    Returns
    -------
        the best model found during the grid search, the best hyperparameters for that model, and the
    performance metrics (accuracy) of the best model on the training, validation, and test sets.
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
    
    best_metrics["train_accuracy"] = accuracy_score(y_train, y_pred_train)
    best_metrics["validation_accuracy"] = accuracy_score(y_valid, y_pred_valid)
    best_metrics["test_accuracy"] = accuracy_score(y_test, y_pred_test)
    
    print("--------------------------------")
    print(f"SKLEARN GRID SEARCH - {model_class}")
    print(f"Best Model: {best_model}")
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


def evaluate_all_models(models_to_run, task_folder):
    '''
    The function `evaluate_all_models` takes a list of models to run and a task folder as input, and for
    each model in the list, it tunes the hyperparameters, saves the model, hyperparameters, and metrics
    in the task folder.

    Parameters
    ----------
    models_to_run
        A list of models to run. It can contain the following values: 'log_reg', 'decision_tree',
        'random_forest', 'gboost'.
    task_folder
        The `task_folder` parameter is a string that specifies the folder where the models and their
        corresponding parameters and metrics will be saved.
    '''

    model_configurations = {
        'log_reg': (LogisticRegression, {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 500, 1000],
            'class_weight': [None, 'balanced']
        }),
        'decision_tree': (DecisionTreeClassifier, {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }),
        'random_forest': (RandomForestClassifier, {
            'n_estimators': [100, 200, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }),
        'gboost': (GradientBoostingClassifier, {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        })
    }

    for model_name in models_to_run:
        model_class, param_grid = model_configurations[model_name]
        
        model, params, metrics = tune_classification_model_hyperparameters(
            model_class, param_grid)
        
        save_model(model, params, metrics, f'{task_folder}/{model_name}/',
                   f'{model_name}_model', f'{model_name}_params', f'{model_name}_metrics')


def find_best_model(task_folder):
    
    '''
    The function `find_best_model` takes a folder path as input, iterates through the models in the
    folder, loads each model, hyperparameters, and metrics, and returns the best model, its
    hyperparameters, and its metrics based on the validation accuracy.
    
    Parameters
    ----------
    task_folder
        The `task_folder` parameter is the path to the folder where the models, hyperparameters, and
    metrics are stored.
    
    Returns
    -------
        The function `find_best_model` returns three values: `best_model`, `best_params`, and
    `best_metrics`.
    '''
    
    best_model = None
    best_params = {}
    best_metrics = {}

    for model_name in os.listdir(task_folder):
        model_folder = os.path.join(task_folder, model_name)

        if not os.path.isdir(model_folder):
            continue

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
        if best_model is None or metrics["validation_accuracy"] > best_metrics["validation_accuracy"]:
            best_model = model
            best_params = hyperparameters
            best_metrics = metrics

    print(best_model, best_params, best_metrics)

    return best_model, best_params, best_metrics


if __name__ == "__main__":
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_split_data(label="bedrooms")
    
    train_logistic_regression()
    
    models = ['log_reg', 'decision_tree', 'random_forest', 'gboost']
    
    evaluate_all_models(models, 'models/classification')
    
    find_best_model('models/classification')