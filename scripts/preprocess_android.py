import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def preprocess_data(data_path, target_column):
    try:
        # Check if the file exists
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
            
        # Read the data
        data = pd.read_csv(data_path)
        print(f"Successfully read data from {data_path}")
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"The dataset at {data_path} is empty.")
            
        # Check if the target column exists
        if target_column not in data.columns:
            raise ValueError(f"The target column '{target_column}' does not exist in the dataset.")
        
        # Identify categorical and numerical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        
        # Handle missing values
        # For numerical columns, impute with mean
        imputer_num = SimpleImputer(strategy='mean')
        data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])
        
        # For categorical columns, impute with most frequent value
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])
        
        # Use LabelEncoder for categorical columns
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col].astype(str))
        
        # Split data into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Convert to float32 to reduce memory usage
        X = X.astype('float32')
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"An error occurred in preprocess_android: {e}")
        raise