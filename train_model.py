import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

# Load data
data = pd.read_csv('data/equipment_anomaly_data.csv')

X = data.drop('faulty', axis=1)
y = data['faulty']

# Separate numerical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model and preprocessor
os.makedirs('models', exist_ok=True)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(pipeline.named_steps['preprocessor'], f)

print(" Model and preprocessor saved.")




