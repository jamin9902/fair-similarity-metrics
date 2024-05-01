import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path):
    # Load the data from CSV
    data = pd.read_csv(file_path)
    
    # Split the data into training and testing sets (80% training, 20% testing)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save the training and testing sets into separate CSV files
    train.to_csv('hire_training_data.csv', index=False)
    test.to_csv('hire_testing_data.csv', index=False)
    
    print("Data has been split and saved to 'hire_training_data.csv' and 'hire_testing_data.csv'.")

split_data('hiring.csv')
