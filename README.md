# InsureAI — Medical Insurance Cost & Risk Predictor

## Business problem
An insurance company wants to estimate customer medical insurance charges and classify customers into normal-cost or high-cost groups.

## Models
- Linear Regression predicts `charges`.
- Logistic Regression predicts `high_cost`.
- R² evaluates regression.
- Accuracy evaluates classification.

The high-cost label is created using the median `charges` in the training dataset: 0 means below the median; 1 means at or above the median. This is a simple project definition, not a medical or insurance-industry standard.

## Setup
1. Put `insurance.csv` in this folder.
2. Open a terminal in this folder.
3. Install libraries: `pip install -r requirements.txt`
4. Train models: `python train_models.py`
5. Start the website: `uvicorn main:app --reload`
6. Open `http://127.0.0.1:8000`

The training script creates `regression_model.joblib`, `classification_model.joblib`, and `threshold.txt`. Do not create these files yourself.
