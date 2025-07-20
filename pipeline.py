import pandas as pd

# Load dataset
df = pd.read_csv("data/equipment_anomaly_data.csv")

# Preview the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check info about columns
print("\nDataFrame Info:")
print(df.info())

import pip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# One-hot encode the categorical features
df_encoded = pd.get_dummies(df, columns=["equipment", "location"])

# Split features and target
X = df_encoded.drop("faulty", axis=1)
y = df_encoded["faulty"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "models/rf_model.pkl")
print("\nModel saved to models/rf_model.pkl")
