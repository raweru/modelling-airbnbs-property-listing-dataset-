# Modelling Airbnb property listing dataset

Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Milestone 1:  Set up the environment

First, we clone the listing images and raw data into a GitHub repository.

## Milestone 2: System Overview

We are about to start developing a framework for evaluating a wide range of machine learning models that can be applied to various datasets. Let's get started by watching an introduction video.

[![Intro Video](video_intro.png)](https://www.youtube.com/watch?v=Tub0xAsNzk8)

## Milestone 3: Data Preparation

Data comes in two folders: tabular data and images. We start by taking a look at tabular data inside listing.csv. There's quite some work to do to clean the data here. We wrote a few functions inside tabular_data.py that take care of:

- Listings with missing ratings.
- "Description" and "Amenities" columns contain lists of strings instead of a string. They also contain newline characters and duplicate phrases that need removing.
- Missing values in "guests", "beds", "bathrooms", "bedrooms".

All data cleaning functions are grouped inside **clean_tabular_data** function inside the __name__ == "__main__" block.

## Milestone 4: Create a Regression Model

After we cleaned the data, we took only numeric data to generate regression models to predict the price per night for each listing. All models and functions were created inside regression.py file and models with their hyperparameters and metrics saved in models/regression/ folder.

We used Regression Models from Scikit-Learn library and compared them to find the best one.

Only data preprocessing that we did on numeric data in the previous step was imputing missing values and feature scaling. So taking it a step further we could have done some feature selection and checking for outliers as well.

Starting point was the default SGD Regressor without any fine tuning. Next two models were the same SGD Regressor but fine tuned using GridSearchCV and a custom fine tune function. We also tried GridSearchCV on a Decision Tree model, Random Forest model and a Gradient Boosting Regressor.

Best performance was achieved with the SGD Regressor with the custom fine tuning function, which gave the lowest value validation RMSE and highest value validation R2.

| Regression Model Name            | Val RMSE          | Val R2                 |
| -------------------------------- | ----------------- | ---------------------- |
| SGD (no fine tuning)             | 0.613             | 0.408                  |
| **SGD + Custom fine tuning**     | **0.583**         | **0.464**              |
| SGD + GridSearchCV               | 0.612             | 0.410                  |
| Decision Tree + GridSearchCV     | 0.672             | 0.289                  |
| Random Forest + GridSearchCV     | 0.642             | 0.350                  |
| Gradient Boosting + GridSearchCV | 0.670             | 0.293                  | 

## Milestone 5: Create a Classification Model

Again, using just the numeric tabular data, we trained classification models to predict which "Category" each listing falls into. All models and functions were created inside classification.py file and models with their hyperparameters and metrics saved in models/classification/ folder.

All models were fine tuned with sklearn GridSearchCV.

| Classification Model Name            | Val Accuracy |
| ------------------------------------ | ------------ |
| Logistic Regression (no fine tuning) | 0.392        |
| Logistic Regression  + GridSearchCV  | 0.408        |
| Decision Tree + GridSearchCV         | 0.272        |
| Random Forest + GridSearchCV         | 0.312        |
| Gradient Boosting + GridSearchCV     | 0.352        |

Best performing was Logistic Regression + GridSearchCV with 0.408 validation accuracy.

