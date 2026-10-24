# InsureAI: FastAPI backend

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="InsureAI")

# Load models created by train_models.py
regression_model = joblib.load("regression_model.joblib")
classification_model = joblib.load("classification_model.joblib")


class Customer(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


def prepare_customer(customer):
    # Use the same number coding that was used during model training
    sex = 0 if customer.sex == "female" else 1
    smoker = 0 if customer.smoker == "no" else 1
    regions = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

    return pd.DataFrame([{
        "age": customer.age,
        "sex": sex,
        "bmi": customer.bmi,
        "children": customer.children,
        "smoker": smoker,
        "region": regions[customer.region]
    }])


@app.get("/")
def homepage():
    return FileResponse("index.html")


@app.post("/predict/regression")
def predict_regression(customer: Customer):
    row = prepare_customer(customer)
    result = float(regression_model.predict(row)[0])
    return {"prediction": round(max(0, result), 2)}


@app.post("/predict/classification")
def predict_classification(customer: Customer):
    row = prepare_customer(customer)
    label = int(classification_model.predict(row)[0])
    probability = float(classification_model.predict_proba(row)[0][1])

    return {
        "prediction": label,
        "result": "High Cost" if label == 1 else "Normal Cost",
        "probability": round(probability * 100, 1)
    }
