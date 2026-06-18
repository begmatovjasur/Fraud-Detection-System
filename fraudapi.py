from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API", description="Bank tranzaksiyalarini tekshirish tizimi")

model = joblib.load('xgboost_fraud_model.pkl')
class TransactionData(BaseModel):
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float

@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    input_data = pd.DataFrame([transaction.model_dump()])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    is_fraud = bool(prediction[0] == 1)
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(probability), 4),
        "message": "Tranzaksiya bloklandi!" if is_fraud else "Tranzaksiya muvaffaqiyatli o'tdi."
    }