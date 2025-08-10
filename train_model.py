import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load Dataset
df = pd.read_csv("dataset/osmi_mental_health_survey.csv")

# 2. Select relevant columns (you may adjust these based on dataset)
features = ["Age", "Gender", "family_history", "work_interfere"]
target = "treatment"  # Whether they sought treatment (proxy for being at risk)

df = df[features + [target]]

# 3. Handle missing values
df.fillna("Unknown", inplace=True)

# 4. Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 5. Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Save model and encoders
joblib.dump(model, "model/mental_health_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("✅ Model training complete and saved!")
