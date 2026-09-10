import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# -----------------------------------------
# Normalize hand landmarks
# -----------------------------------------
def normalize_landmarks(row):

    landmarks = np.array(row, dtype=float).reshape(21, 3)

    # Wrist = landmark 0
    wrist = landmarks[0]

    # Make wrist the origin
    landmarks = landmarks - wrist

    # Find maximum distance from wrist
    distances = np.linalg.norm(landmarks, axis=1)
    scale = np.max(distances)

    # Avoid division by zero
    if scale > 0:
        landmarks = landmarks / scale

    # Convert back to 63 values
    return landmarks.flatten()


# -----------------------------------------
# Load dataset
# -----------------------------------------
data = pd.read_csv("sign_data.csv")

print("Dataset shape:", data.shape)

print("\nSigns:")
print(data["label"].value_counts())


# -----------------------------------------
# Separate features and labels
# -----------------------------------------
X_raw = data.drop("label", axis=1)
y = data["label"]


# -----------------------------------------
# Normalize every sample
# -----------------------------------------
X = np.array([
    normalize_landmarks(row)
    for row in X_raw.values
])


# -----------------------------------------
# Train/test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# Create Random Forest
# -----------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# -----------------------------------------
# Train
# -----------------------------------------
print("\nTraining model...")

model.fit(X_train, y_train)


# -----------------------------------------
# Test
# -----------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------------
# Save model
# -----------------------------------------
joblib.dump(model, "sign_model.pkl")

print("\nNormalized model saved as sign_model.pkl ✅")