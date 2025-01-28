import os
import joblib
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_dataset(dataset_file, target_variable):
    try:
        # Load the dataset with low_memory=False to handle mixed types
        df = pd.read_csv(dataset_file, low_memory=False)
        
        # Check if the target variable exists
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in dataset.")
        
        # Split into features and labels
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load the saved scaler for this dataset
        scaler_filename = os.path.splitext(dataset_file)[0] + '_scaler.pkl'
        scaler = joblib.load(scaler_filename)
        
        # Apply scaling to the features
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    
    except Exception as e:
        print(f"Error loading or preprocessing dataset {dataset_file}: {str(e)}")
        return None, None

def evaluate_model(model, X_test, y_test, attack_type):
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix for {attack_type}")
    plt.savefig(f"results/confusion_matrix_{attack_type}.png")
    plt.close()
    
    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.title(f"ROC Curve for {attack_type}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"results/roc_curve_{attack_type}.png")
    plt.close()
    
    # Metrics
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    
    return precision, accuracy, sensitivity

# Mapping from attack type to dataset and target variable
model_dataset_map = {
    'android': ('../data/android_malware_data.csv', 'Label'),
    'ddos': ('../data/ddos_data.csv', 'Label'),
    'phishing': ('../data/phishing_data.csv', 'CLASS_LABEL')
}

# Create a results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# List of model paths
model_paths = [
    "models/android_logisticregression.pkl",
    "models/android_randomforest.pkl",
    "models/ddos_logisticregression.pkl",
    "models/ddos_randomforest.pkl",
    "models/phishing_logisticregression.pkl",
    "models/phishing_randomforest.pkl"
]

# Dictionary to store results
results = {
    "Attack Type": [],
    "Model Type": [],
    "Precision": [],
    "Accuracy": [],
    "Sensitivity": []
}

# Evaluate each model
for model_path in model_paths:
    # Extract attack type and model type from the filename
    attack_type = os.path.basename(model_path).split("_")[0]
    model_type = os.path.basename(model_path).split("_")[1].split(".")[0]
    
    # Get the corresponding dataset and target variable
    dataset_file, target_variable = model_dataset_map.get(attack_type, (None, None))
    
    if dataset_file is None or target_variable is None:
        print(f"Error: No dataset mapping found for attack type '{attack_type}'")
        continue
    
    # Load and preprocess the dataset
    X_test, y_test = load_and_preprocess_dataset(dataset_file, target_variable)
    
    if X_test is None or y_test is None:
        print(f"Error: Failed to load or preprocess dataset for {attack_type}")
        continue
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Evaluate the model
        precision, accuracy, sensitivity = evaluate_model(model, X_test, y_test, attack_type)
        
        # Append results to the dictionary
        results["Attack Type"].append(attack_type)
        results["Model Type"].append(model_type)
        results["Precision"].append(precision)
        results["Accuracy"].append(accuracy)
        results["Sensitivity"].append(sensitivity)
        
        print(f"Evaluation completed for {attack_type} - {model_type}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error evaluating {model_path}: {str(e)}")
        print("-" * 50)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_evaluation_results.csv", index=False)
print("Results saved to results/model_evaluation_results.csv")