from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from modelling import save_model
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

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


def evaluate_all_models(models_to_run):

    if 'log_reg_base' in models_to_run:
        
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
        
        save_model(log_reg_model, log_reg_params, log_reg_metrics, 'models/classification/log_reg_base/',
                   'log_reg_model', 'log_reg_params', 'log_reg_metrics')


if __name__ == "__main__":
    
    train_logistic_regression()
    
    models_to_run = ['log_reg_base']
    
    evaluate_all_models(models_to_run)