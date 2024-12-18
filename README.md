## Property Price Prediction Project
This project aims to predict property prices based on various features such as the number of rooms, location, population, and more. The project is divided into multiple components, including data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Project Structure
The project is organized as follows:

/Property_Price_Prediction/                                                                                                                                                                                              
├── notebooks/                                                                                                                  
│   ├── eda.ipynb                                                                                                                
│   ├── property_price_prediction.ipynb                                                                                         
├── property_price_prediction/                                                                                                  
│   ├── preprocess.py                                                                                                           
│   ├── EDA.py                                                                                                                  
│   ├── train.py                                                                                                                
│   ├── evaluate.py  
│   ├──  main.py                                                                                                               
│   ├── eda_images/                                                                                                             
└── README.md   

## Notebooks
eda.ipynb:
This notebook is used for Exploratory Data Analysis (EDA). It explores the dataset by visualizing distributions and relationships between variables. It also performs feature engineering and prepares the data for model training.

property_price_prediction.ipynb:
This notebook contains all the steps needed for the property price prediction process:

It loads the preprocessed data.
It trains the models (simple linear regression and multiple linear regression).
It evaluates and compares the models using various evaluation metrics (e.g., R² score, MAE, RMSE).

## Python Files
preprocess.py:
This file contains the function load_and_preprocess_data which:

Loads the data from the CSV file.
Fills missing values (specifically for the 'total_bedrooms' column).
One-hot encodes the 'ocean_proximity' column using pd.get_dummies.
Performs feature engineering by creating new features like 'rooms_per_household', 'bedrooms_per_room', and 'population_per_household'.

EDA.py:
This file defines the eda function which generates and saves visualizations of the data in the eda_images/ folder. The EDA includes:

A histogram for the distribution of 'median_house_value'.
A scatter plot for 'median_income' vs 'median_house_value'.
A heatmap of the correlation matrix for the features.
train.py:
This file contains the train_models function which trains two models:

A simple linear regression model using 'median_income' as the only feature.
A multiple linear regression model using a combination of top 5 important features. It then returns the trained models, test data, and target values for evaluation.
evaluate.py:
This file contains the evaluate_model function that calculates and returns evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score for each model.

The print_evaluation_results function prints the evaluation results in a readable format.

eda_images/:
This folder stores the images generated by the EDA process. It includes visualizations such as:

Distribution of the median house value.
Scatter plot of 'median_income' vs 'median_house_value'.
Correlation matrix heatmap.
Main Workflow (in main.py)
The main.py file orchestrates the following steps:

Preprocess the Data:

Loads and preprocesses the dataset using the load_and_preprocess_data function from preprocess.py.
Perform EDA:

Performs Exploratory Data Analysis using the eda function from EDA.py, generating and saving visualizations in the eda_images/ folder.
Train the Models:

Trains two models using the train_models function from train.py: one for simple linear regression and one for multiple linear regression.
Evaluate the Models:

Evaluates the models using the evaluate_model function from evaluate.py and prints the evaluation metrics using print_evaluation_results.

import pandas as pd
from preprocess import load_and_preprocess_data
from EDA import eda
from train import train_models
from evaluate import evaluate_model, print_evaluation_results

# Step 1: Preprocess the Data
data_path ='../data/data_file.csv'
data = load_and_preprocess_data(data_path)

# Step 2: Perform EDA
eda(data)

# Step 3: Train the Models
model_simple, model_multiple, X_test_simple, X_test_multiple, y_test = train_models(data)

# Step 4: Evaluate the Models
simple_metrics = evaluate_model(model_simple, X_test_simple, y_test)
multiple_metrics = evaluate_model(model_multiple, X_test_multiple, y_test)

# Print Evaluation Results
print_evaluation_results("Simple Linear Regression", simple_metrics)
print_evaluation_results("Multiple Linear Regression", multiple_metrics)
Model Details
Simple Linear Regression Model:

This model uses only 'median_income' as the predictor to estimate 'median_house_value'.
Evaluation metrics include:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score (coefficient of determination)


Multiple Linear Regression Model:

This model uses a combination of the following features: 'median_income', 'latitude', 'total_rooms', 'housing_median_age', and 'ocean_proximity_INLAND'.
Evaluation metrics are the same as for the simple linear regression model.
Feature Engineering & Data Processing
Handling Missing Values:

The missing values in the 'total_bedrooms' column are filled with the mean value of the column.
One-Hot Encoding:

The 'ocean_proximity' column is one-hot encoded to convert categorical values into binary columns using pd.get_dummies.
Feature Engineering:

New features are created to better capture the relationships in the data:
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
Conclusion
This project demonstrates how to predict property prices based on available features. It uses both simple and multiple linear regression models and evaluates their performance using common metrics like MAE, MSE, RMSE, and R². The process includes handling missing values, encoding categorical features, and creating new features to improve model accuracy.

