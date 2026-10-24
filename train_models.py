# InsureAI: train one regression model and one classification model

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

# Read insurance.csv from this same folder
data = pd.read_csv("insurance.csv")

# Change text categories into numbers
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data["region"] = data["region"].map({
    "northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3
})

# These six columns are the input information for both models
features = ["age", "sex", "bmi", "children", "smoker", "region"]

# ---------------- REGRESSION ----------------
X = data[features]
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

predicted_charges = regression_model.predict(X_test)
print("Regression R2 score:", round(r2_score(y_test, predicted_charges), 3))

joblib.dump(regression_model, "regression_model.joblib")


# ---------------- CLASSIFICATION ----------------
# Median charge is the simple rule for defining the high-cost group.
# 0 = normal cost; 1 = high cost.
threshold = data["charges"].median()
data["high_cost"] = (data["charges"] >= threshold).astype(int)

X = data[features]
y = data["high_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classification_model = LogisticRegression(max_iter=1000)
classification_model.fit(X_train, y_train)

predicted_group = classification_model.predict(X_test)
print("Classification accuracy:", round(accuracy_score(y_test, predicted_group), 3))

joblib.dump(classification_model, "classification_model.joblib")

with open("threshold.txt", "w") as file:
    file.write(str(threshold))

print("Both models saved successfully.")
print("High-cost threshold:", round(threshold, 2))
