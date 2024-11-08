import pandas as pd

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Fill missing values in 'total_bedrooms' with the column mean
    data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
    
    # One-hot encode 'ocean_proximity' column
    data_encoded = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
    
    # Feature Engineering
    data_encoded['rooms_per_household'] = data_encoded['total_rooms'] / data_encoded['households']
    data_encoded['bedrooms_per_room'] = data_encoded['total_bedrooms'] / data_encoded['total_rooms']
    data_encoded['population_per_household'] = data_encoded['population'] / data_encoded['households']
    
    return data_encoded
