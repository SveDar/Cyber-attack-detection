import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(filepath, target_column):
    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Handle missing values
    data = data.dropna()
    print(f"Data after dropping missing values. Shape: {data.shape}")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features normalized.")

    # Apply PCA (retain 95% variance)
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"Features after PCA. Shape: {X_reduced.shape}")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    print(f"Data split into training and testing sets.")

    return X_train, X_test, y_train, y_test

# Preprocess Phishing data
if __name__ == "__main__":
    phishing_data_path = "../ata/phishing_data.csv"  # Update path
    target_column = "CLASS_LABEL"  # Corrected target column name

    print("Preprocessing Phishing data...")
    X_train, X_test, y_train, y_test = preprocess_data(phishing_data_path, target_column)
    print("Phishing preprocessing completed!")
