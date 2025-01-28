import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess_ddos import preprocess_data as preprocess_ddos
from preprocess_phishing import preprocess_data as preprocess_phishing
from preprocess_android import preprocess_data as preprocess_android
#from preprocess_sqli import preprocess_data as preprocess_sqli

# Select the dataset to train on
DATASET = "Android"  # Choose from "DDoS", "Phishing", "SQLI", "Android"
ALGORITHM = "LogisticRegression"  # Choose from "RandomForest", "LogisticRegression"

# Load the correct dataset
if DATASET == "DDoS":
    data_path = "../data/ddos_data.csv"
    target_column = "Label"
    X_train, X_test, y_train, y_test = preprocess_ddos(data_path, target_column)

elif DATASET == "Phishing":
    data_path = "../data/phishing_data.csv"
    target_column = "LABEL"
    X_train, X_test, y_train, y_test = preprocess_phishing(data_path, target_column)

elif DATASET == "Android":
    data_path = "../data/android_malware_data.csv"
    target_column = "Label"
    X_train, X_test, y_train, y_test = preprocess_android(data_path, target_column)

elif DATASET == "SQLI":
    data_path = "../data/sqli_data.csv"
    target_column = "Label"
    X_train, X_test, y_train, y_test, vectorizer = preprocess_sqli(data_path, target_column)  # Fixing the commented-out part

# Train the model
def train_model(X_train, y_train, algorithm):
    if algorithm == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif algorithm == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    
    print(f"Training {algorithm} model on {DATASET} dataset...")
    model.fit(X_train, y_train)
    return model

# Save model and vectorizer (for SQLI dataset)
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    model = train_model(X_train, y_train, ALGORITHM)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    model_filename = f"models/{DATASET.lower()}_{ALGORITHM.lower()}.pkl"
    save_model(model, model_filename)
