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

