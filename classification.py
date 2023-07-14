from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from regression import save_model
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
import json
import os
import joblib


# Ignore the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ignore the data conversion warning
warnings.filterwarnings("ignore", category=DataConversionWarning)


# Load the tabular data using load_airbnb function
features, labels = load_airbnb(label="Category")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


def train_logistic_regression():
    
    # Initialize and train the linear regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Compute predictions for the training set
    train_predictions = model.predict(X_train)
    
    # Compute predictions for the validation set
    valid_predictions = model.predict(X_valid)

    # Compute predictions for the test set
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

    if 'log_reg' in models_to_run:
        
        model_class = LogisticRegression
        
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 500, 1000],
            'class_weight': [None, 'balanced']
            }
        
        log_reg_model, log_reg_params, log_reg_metrics = tune_classification_model_hyperparameters(
            model_class, param_grid)
        
        save_model(log_reg_model, log_reg_params, log_reg_metrics, f'{task_folder}/log_reg/',
                   'log_reg_model', 'log_reg_params', 'log_reg_metrics')

    if 'decision_tree' in models_to_run:
        
        model_class = DecisionTreeClassifier
        
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
            }
        
        dec_tree_model, dec_tree_params, dec_tree_metrics = tune_classification_model_hyperparameters(
            model_class, param_grid)
        
        save_model(dec_tree_model, dec_tree_params, dec_tree_metrics, f'{task_folder}/dec_tree/',
                   'dec_tree_model', 'dec_tree_params', 'dec_tree_metrics')
        
    if 'random_forest' in models_to_run:
        
        model_class = RandomForestClassifier
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
            }
        
        rnd_for_model, rnd_for_params, rnd_for_metrics = tune_classification_model_hyperparameters(
            model_class, param_grid)
        
        save_model(rnd_for_model, rnd_for_params, rnd_for_metrics, f'{task_folder}/rnd_for/',
                   'rnd_for_model', 'rnd_for_params', 'rnd_for_metrics')
        
    if 'gboost' in models_to_run:
        
        model_class = GradientBoostingClassifier
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
            }
        
        gboost_model, gboost_params, gboost_metrics = tune_classification_model_hyperparameters(
            model_class, param_grid)
        
        save_model(gboost_model, gboost_params, gboost_metrics, f'{task_folder}/gboost/',
                   'gboost_model', 'gboost_params', 'gboost_metrics')

def find_best_model(task_folder):
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
    
    train_logistic_regression()
    
    models = ['log_reg', 'decision_tree', 'random_forest', 'gboost']
    
    #evaluate_all_models(models, 'models/classification')
    
    find_best_model('models/classification')