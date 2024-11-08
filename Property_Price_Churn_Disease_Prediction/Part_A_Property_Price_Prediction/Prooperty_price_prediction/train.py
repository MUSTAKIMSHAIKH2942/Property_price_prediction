from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_models(data, target='median_house_value'):
    # Define target and features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train simple linear regression model (using 'median_income' as a single feature)
    X_train_simple = X_train[['median_income']]
    X_test_simple = X_test[['median_income']]
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train)
    
    # Train multiple linear regression model (using selected features)
    features = ['median_income', 'latitude', 'total_rooms', 'housing_median_age', 'ocean_proximity_INLAND']
    X_train_multiple = X_train[features]
    X_test_multiple = X_test[features]
    model_multiple = LinearRegression()
    model_multiple.fit(X_train_multiple, y_train)
    
    # Save the models
    joblib.dump(model_simple, 'models/simple_linear_regression_model.pkl')
    joblib.dump(model_multiple, 'models/multiple_linear_regression_model.pkl')
    
    return model_simple, model_multiple, X_test_simple, X_test_multiple, y_test
