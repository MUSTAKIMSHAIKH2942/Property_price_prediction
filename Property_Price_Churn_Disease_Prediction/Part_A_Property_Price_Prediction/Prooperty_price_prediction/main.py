import pandas as pd
from preprocess import load_and_preprocess_data
from EDA import eda
from train import train_models
from evaluate import evaluate_model, print_evaluation_results

# Step 1: Preprocess the Data
data_path = '../data/data_file.csv'
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
