# Modelling Airbnb property listing dataset

Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Milestone 1:  Set up the environment

First, we clone the listing images and raw data into a GitHub repository. This step ensures consistent data availability and allows for easy sharing and collaboration.

## Milestone 2: System Overview

We are about to start developing a framework for evaluating a wide range of machine learning models that can be applied to various datasets. Let's get started by watching an introduction video.

[![Intro Video](video_intro.png)](https://www.youtube.com/watch?v=Tub0xAsNzk8)

- Using data to train machine learning models can lead to many opportunities.
- There are various ways to model data, including linear models, support vector machines, non-parametric models, decision trees, random forests, xg boost, light gbm, and neural networks.
- Convolutional neural networks are used for image processing, while transformer models like BERT are used for text processing.
- With embeddings, all models can be used to process text.
- Multimodal models can process multiple modalities of data at once.
- Evaluating a range of models is necessary to find the best approach in each situation.
- Building frameworks and platforms can make it easier to train and evaluate machine learning models.
- The challenge for companies is finding people who can build these platforms and have the intuition for what will work.

## Milestone 3: Data Preparation

In this milestone, the raw Airbnb property listing data undergoes a series of cleaning and transformation steps to ensure its suitability for analysis and model training. The data preparation process primarily focuses on the tabular data component of the dataset. Below, we provide an overview of the key functions and steps used to achieve this goal.

### Data Cleaning and Transformation Functions

#### `remove_rows_with_missing_ratings(listings)`

This function addresses missing values in the "Value_rating" column by removing rows that lack valid ratings. It accepts a DataFrame containing listing data and returns an updated DataFrame with rows containing missing ratings removed.

#### `combine_description_strings(listings)`

The `combine_description_strings` function refines the "Description" column by performing various operations, such as removing specific listing descriptions, converting strings of lists to actual lists, and eliminating unwanted elements. The result is an enhanced "Description" column that provides cleaner and more concise information about each listing.

#### `combine_amenities_strings(listings)`

The `combine_amenities_strings` function processes the "Amenities" column by converting strings of lists to actual lists, removing unnecessary elements, and reformatting the data for improved readability. The transformed "Amenities" column provides a consolidated view of the amenities offered by each listing.

#### `set_default_feature_values(listings)`

To address missing values in essential features, the `set_default_feature_values` function fills in default values for the "guests," "beds," "bathrooms," and "bedrooms" features. This ensures that the dataset remains complete and consistent.

#### `replace_newlines(text)`

The `replace_newlines` function replaces consecutive newlines in text with either a dot and a space or a single space, depending on the context. This enhances the readability and structure of the text data.

### Standardization

After data cleaning, the continuous numeric features undergo standardization to bring them to a common scale. The StandardScaler from Scikit-Learn is utilized for this purpose.

### Label Encoding

For classification tasks involving categorical labels, the "Category" labels are encoded using the LabelEncoder. This converts categorical labels into numerical values for compatibility with machine learning algorithms.

### Loading the Cleaned Data

The final cleaned and preprocessed tabular data is saved as a CSV file named "clean_tabular_data.csv." Additionally, the `load_airbnb` function is provided to load this data for further analysis and model training. The function allows for flexible loading of features and labels, catering to specific use cases.

## Milestone 4: Fine-Tuning Our Regression Models

Welcome to the fourth and pivotal phase of our analysis, where we take a closer look at refining our regression models to achieve better accuracy. Our main goals here are to enhance our models' prediction capabilities and gain insights into how different settings affect their performance.

### Personalized Model Enhancement: A Deeper Dive

Our journey begins with a tailored approach called *custom_tune_regression_model_hyperparameters*. We use this approach specifically for the *SGDRegressor* model. To achieve this, we explore a range of hyperparameter settings, like adjusting knobs on a radio to find the best frequency. Here's how we set up the adjustments using a simple dictionary:

```python
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
```

### Methodical Model Exploration with Scikit-Learn

Expanding our toolkit, we now employ the *tune_regression_model_hyperparameters* function, which partners with Scikit-Learn's *GridSearchCV* to help us fine-tune multiple regression models. We focus on models like *DecisionTreeRegressor*, *RandomForestRegressor*, and *GradientBoostingRegressor*. For this, we use a dictionary that defines various settings:

```python
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
```

This dictionary guides us in finding the perfect combination of settings to capture the essence of your data accurately.

### Thorough Evaluation and Model Safekeeping

With a suite of models in our arsenal, we turn to the *evaluate_all_models* function to provide a comprehensive assessment.This rigorous process helps us understand the strengths and limitations of each model, enabling us to make well-informed choices.

To ensure that our best models are well-preserved, we use methods like *save_model*. These steps ensure that the hard work put into refining the models remains intact and ready for future use.

Best performance was achieved with the SGD Regressor with the custom fine tuning function, which gave the lowest validation RMSE value and highest validation R_squared value.

| Regression Model Name            | Val RMSE          | Val R_squared          |
| -------------------------------- | ----------------- | ---------------------- |
| SGD (no fine tuning)             | 80.74             | 0.384                  |
| **SGD + Custom fine tuning**     | **78.93**         | **0.412**              |
| SGD + GridSearchCV               | 81.26             | 0.376                  |
| Decision Tree + GridSearchCV     | 86.77             | 0.289                  |
| Random Forest + GridSearchCV     | 81.56             | 0.372                  |
| Gradient Boosting + GridSearchCV | 85.61             | 0.308                  |

## Milestone 5: Create a Classification Model

Again, using just the numeric tabular data, we trained classification models to predict which "Category" each listing falls into. All models and functions were created inside classification.py file and models with their hyperparameters and metrics saved in models/classification/ folder.

## Milestone 5: Crafting Accurate Classification Models

Welcome to the fifth and exciting phase of our journey, where we delve into creating classification models. In this phase, we build models that can categorize data into different groups, helping you make insightful predictions and informed decisions.

### Understanding Classification Models

Just as we refined regression models earlier, our focus now shifts to crafting effective classification models. Let's explore the key classification models we're using, along with their strengths and limitations:

#### Logistic Regression: Versatile and Understandable

We start with the *Logistic Regression* model, a versatile and understandable choice. It's great for tasks where we need to classify things into two categories. We can adjust its performance using settings like regularization strength and solver algorithms. Here's an example of the settings we use:

```python
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 500, 1000],
    'class_weight': [None, 'balanced']
}
```

Why Logistic Regression? It's a simple and interpretable model that works well when you have a clear boundary between categories.

#### Decision Tree: Unveiling Patterns

Next up is the Decision Tree classifier, which uncovers patterns based on features. It can capture complex relationships in data. We adjust settings like tree depth and split criteria to fit the data better. Here's an example of the settings:

```python
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}
```

Why Decision Tree? It's useful when there are non-linear relationships in data and can handle different kinds of categories.

#### Random Forest: Strength in Numbers

Now, let's introduce the Random Forest classifier. It's like a team of decision trees working together. With settings for the number of trees, depth, and more, we create a strong ensemble. Here's an example of the settings:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}
```

Why Random Forest? It's excellent for improving accuracy and handling various types of data, reducing overfitting.

#### Gradient Boosting: Boosting Performance

Lastly, we have the Gradient Boosting classifier. It boosts performance by learning from errors. With settings for boosting stages, learning rate, and more, we enhance its accuracy. Here's an example of the settings:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}
```

Why Gradient Boosting? It combines multiple models to excel in complex tasks, yielding high accuracy.

### Methodical Evaluation and Safekeeping

To ensure our classification models are trustworthy, we evaluate them using the evaluate_all_models function. Similar to before, this step helps us make informed decisions about model selection.

We also save the best models, settings, and metrics using the save_model function, securing them for future use.

All models were fine tuned with sklearn GridSearchCV.

| Classification Model Name            | Val Accuracy |
| ------------------------------------ | ------------ |
| Logistic Regression (no fine tuning) | 0.392        |
| **Logistic Regression  + GridSearchCV**  | **0.408**        |
| Decision Tree + GridSearchCV         | 0.320        |
| Random Forest + GridSearchCV         | 0.344        |
| Gradient Boosting + GridSearchCV     | 0.344        |

Best performing was Logistic Regression + GridSearchCV with 0.408 validation accuracy.

## Milestone 6: Create a configurable neural network

Welcome to the exciting realm of neural networks! In this milestone, we've pushed the boundaries of our regression problem by creating a configurable neural network. This allows you to easily customize and fine-tune the neural network's performance by tweaking parameters in the `nn_config.yaml` file. Let's dive into the details of our journey.

### Designing the Configurable Neural Network

Our neural network architecture is designed to predict nightly listing prices effectively. It's crafted with layers of neurons that process the input data, gradually learning to make accurate predictions. The key components of our architecture include:

- **Input Layer:** This layer accepts the input features, in our case, the listing information, to initiate the prediction process.
- **Hidden Layers:** These layers contain neurons that process and transform the input features using a defined activation function. The number of hidden layers and the width of each layer can be adjusted using the `hidden_layer_width` parameter.
- **Output Layer:** The final layer produces the predicted nightly listing price.

### Hyperparameters Customization

We understand that every problem requires different tuning. That's why we introduced the `nn_config.yaml` file, allowing you to customize hyperparameters such as learning rate, hidden layer width, and the number of training epochs. This empowers you to adapt the neural network to your specific needs, optimizing its performance for your dataset.

### Training and Visualization

After defining the architecture and configuring the hyperparameters, we trained the neural network using the provided training, validation, and test datasets. We utilized the powerful PyTorch framework to efficiently perform gradient-based optimization and improve the model's predictive capabilities.

We used the `SummaryWriter` class from PyTorch's `tensorboard` package to visualize the training process using Tensorboard. This visual aid helps you monitor the training's progress, ensuring that you are on the right track.

### Fine-Tuning for Optimal Performance

To achieve the best possible model, we fine-tuned the neural network using different configuration combinations. We aimed to minimize the root mean squared error (RMSE) and maximize the coefficient of determination (R-squared) on the validation dataset. This process allowed us to select the model that demonstrated the highest predictive accuracy.

### Showcasing the Best Model

Our efforts in fine-tuning led us to the best-performing model, which achieved a validation RMSE_loss of 69.24 and an R_squared value of 0.425. These metrics demonstrate the neural network's ability to accurately predict nightly listing prices based on the provided features.

![Tensorboard](tensorboard.png)

### Model Safekeeping

As with our previous models, we took care to ensure that our hard work is preserved. We saved the best neural network model, its hyperparameters, and metrics to a designated folder. This ensures that the fruits of our labor remain accessible and ready for further analysis and deployment.

With this configurable neural network, you can confidently tackle predictive tasks, harnessing the power of artificial intelligence to extract insights and make informed decisions. Your journey into the world of neural networks has just begun, and the possibilities are limitless!

<img src="nn.jpg" alt="Tensorboard Screenshot" width="500">

## Milestone 7: Reuse the framework for another use-case with the Airbnb data

For our final endeavor, we repurposed our classification shallow algorithms and adapted our neural network for a classification problem. In this task, we aimed to predict the number of bedrooms ('bedrooms' column) as our label.

### Shallow Algorithm Reuse

Repurposing the classification shallow algorithms from `classification.py` was a breeze. We achieved this by merely replacing the 'label' keyword argument when loading the dataset.

### Tweaking the Neural Network

Adapting the neural network for classification required a few tweaks:

- **Loss Function:** We switched to using `nn.CrossEntropyLoss` as the loss function, fitting the classification context.
- **Evaluation Metrics:** The evaluation metrics were overhauled to focus on accuracy, precision, recall, and F1-score.

Our classification neural network was neatly saved into `neural_net_class.py`.

### Model Performance Summary

Here's a snapshot of the performance of different classification models:

| Classification Model Name            | Validation Accuracy |
| ------------------------------------ | ------------------ |
| Logistic Regression (no fine tuning) | 0.704              |
| Logistic Regression + GridSearchCV   | 0.728              |
| Decision Tree + GridSearchCV         | 0.760              |
| **Random Forest + GridSearchCV**     | **0.808**          |
| Gradient Boosting + GridSearchCV     | 0.784              |
| MLP Classifier                       | 0.792              |

#### RandomForestClassifier: The Star Performer

Among the shallow algorithms, the RandomForestClassifier truly stood out, achieving the highest validation accuracy of 0.808.

**Best Hyperparameters:**

- Criterion: gini
- Max_depth: 10
- Max_features: sqrt
- Min_samples_leaf: 1
- Min_samples_split: 5
- N_estimators: 100

**Best Metrics:**

| Metric              | Train Accuracy | Validation Accuracy | Test Accuracy |
|---------------------|----------------|---------------------|---------------|
| Accuracy            | 0.967          | 0.808               | 0.782         |

#### MLP Classifier: A Close Contender

The MLP Classifier provided a slightly lower validation accuracy of 0.792.

**Best Hyperparameters:**

- Learning Rate: 0.001
- Hidden Layer Width: [100, 25]
- Num_epochs: 100
- Depth: 2

**Best Metrics:**

| Metric                 | Training | Validation | Test      |
|------------------------|----------|------------|-----------|
| Accuracy               | 0.774    | 0.792      | 0.774     |
| Inference Latency      | 3.196e-05| 3.214e-05  | 3.225e-05 |
| CrossEntropyLoss       | 0.702    | 0.715      | 0.742     |

### Visualization: MLP Classifier Performance

Here's a visual representation of the MLP Classifier's training and validation accuracy using Tensorboard:

![MLP Classifier Performance](nn_class.png)

With these insightful results, we conclude our journey of repurposing and fine-tuning for classification. The RandomForestClassifier emerged as the champion, showcasing its prowess in classifying the number of bedrooms accurately. Our MLP Classifier put up a strong fight, falling just slightly short of the RandomForest's performance.

In this comprehensive journey, we've covered the spectrum of data analysis, from preprocessing and exploration to fine-tuning and repurposing models. Armed with a robust set of regression and classification tools, we've harnessed the power of machine learning to make accurate predictions and insightful classifications. With these results at hand, we're now equipped to drive analyses and make informed decisions across a multitude of domains.

As we wrap up our journey through model development and analysis, we're excited about the possibilities ahead. One exciting direction to explore is combining text and tabular data to create even more powerful models.