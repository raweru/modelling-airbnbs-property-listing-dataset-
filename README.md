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

After we cleaned the data, we took only numeric data to generate regression models to predict the price per night for each listing. All models and functions were created inside modelling.py file and models with their hyperparameters and metrics saved in models/regression/ folder.

We used Regression Models from Scikit-Learn library to find the best one. 

Only data preprocessing that we did on numeric data in the previous step was imputing missing values. So taking it a step further we could have done some feature scaling, selection and check for outliers.

Starting point was the default SGD Regressor without any fine tuning. Next two models were the same SGD Regressor but fine tuned using GridSearchCV and a custom fine tune function. We also tried GridSearchCV on a Decision Tree model, Random Forest model and a Gradient Boosting Regressor.

Best performance was achieved with the Random Forest Regressor, which gave the lowest value validation RMSE and highest value validation R2.

| Regression Model Name            | Val RMSE          | Val R2                 |
| -------------------------------- | ----------------- | ---------------------- |
| SGD (no fine tuning)             | 19393676086.58724 | -3.548103508824183e+16 |
| SGD + Custom fine tuning         | 83.55228726659075 | 0.3414440283871991     |
| SGD + GridSearchCV               | 86.57500452693962 | 0.2929322137573135     |
| Decision Tree + GridSearchCV     | 91.37535559822163 | 0.2123483888665909     |
| Random Forest + GridSearchCV     | 82.21885802999995 | 0.3622963741081465     |
| Gradient Boosting + GridSearchCV | 86.57975827578649 | 0.2928545628239384     | 

