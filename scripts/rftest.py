import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
df = pd.read_csv('../data/phishing_data.csv')

# Step 2: Separate features and target
X = df.drop('CLASS_LABEL', axis=1)  # Features
y = df['CLASS_LABEL']  # Target

# Step 3: Check for missing values (if any)
print("Missing values in each column:")
print(X.isnull().sum())

# Step 4: Encode categorical features (if any)
# In this dataset, all features appear to be numerical, so no encoding is needed.

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Calculate feature importance
feature_importance = model.feature_importances_

# Create a DataFrame to visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print("Feature Importance:")
print(feature_importance_df)

# Step 8: Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Step 9: Identify unnecessary columns
# Set a threshold for importance (e.g., 0.01)
threshold = 0.01

# Identify unnecessary columns
unnecessary_columns = feature_importance_df[feature_importance_df['Importance'] < threshold]['Feature'].tolist()
print("Unnecessary columns (importance < threshold):", unnecessary_columns)

# Step 10: Remove unnecessary columns
X_filtered = X.drop(columns=unnecessary_columns)
print("Remaining columns after removing unnecessary ones:")
print(X_filtered.columns)