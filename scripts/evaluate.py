import joblib
from sklearn.metrics import accuracy_score, classification_report
from preprocess_ddos import preprocess_data as preprocess_ddos
from preprocess_phishing import preprocess_data as preprocess_phishing
from preprocess_android import preprocess_data as preprocess_android

# Define the dataset and algorithm to evaluate
DATASET = "Phishing"  # Choose from "DDoS", "Phishing", "Android"
ALGORITHM = "RandomForest"  # Choose from "RandomForest", "LogisticRegression"

# Load the corresponding test dataset
if DATASET == "DDoS":
    test_data_path = "../data/test_data/ddos_data_test.csv"
    target_column = "Label"
    # Preprocess the test data
    # Assuming preprocess_ddos returns X_train, X_test, y_train, y_test, scaler
    _, X_test, _, y_test, _ = preprocess_ddos(test_data_path, target_column)

elif DATASET == "Phishing":
    test_data_path = "../data/test_data/phishing_data_test.csv"
    target_column = "CLASS_LABEL"
    # Preprocess the test data without loading saved preprocessing objects
    _, X_test, _, y_test = preprocess_phishing(test_data_path, target_column)

elif DATASET == "Android":
    test_data_path = "../data/test_data/android_malware_data_test.csv"
    target_column = "Label"
    # Preprocess the test data
    # Assuming preprocess_android returns X_train, X_test, y_train, y_test
    _, X_test, _, y_test = preprocess_android(test_data_path, target_column)

# Load the trained model
model_filename = f"models/{DATASET.lower()}_{ALGORITHM.lower()}.pkl"
model = joblib.load(model_filename)

# Evaluate the model on the test data
print(f"Evaluating {ALGORITHM} model on {DATASET} test dataset...")
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Model Performance on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))