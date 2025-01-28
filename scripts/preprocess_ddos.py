import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data(filepath, target_column):
    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath)

    # Drop non-numeric columns (IP addresses and timestamps)
    data = data.drop(columns=['ip.src', 'ip.dst', 'frame.time'], errors='ignore')

    # Handle missing values
    data = data.dropna()
    
    # Encode target labels
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction (optional)
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X_scaled)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    print("Preprocessing completed successfully.")
    return X_train, X_test, y_train, y_test

# Preprocess DDoS data
if __name__ == "__main__":
    ddos_data_path = "../data/ddos_data.csv"  # Update path
    target_column = "Label"  # Ensure this is the correct column name

    print("Starting preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(ddos_data_path, target_column)
    print("DDoS preprocessing completed!")
