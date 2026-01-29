import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load CSV dataset (NO headers in file)
columns = [
    'cultivar',
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols',
    'flavanoids',
    'nonflavanoid_phenols',
    'proanthocyanins',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

data = pd.read_csv("wine.csv")

# convert cultivar to integer (safety)
data['cultivar'] = data['cultivar'].astype(int)

X = data[
    [
        'alcohol',
        'malic_acid',
        'alcalinity_of_ash',
        'magnesium',
        'flavanoids',
        'color_intensity'
    ]
]

y = data['cultivar'] - 1


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling (MANDATORY)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save Model & Scaler
joblib.dump(model, "model/wine_cultivar_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved successfully!")
