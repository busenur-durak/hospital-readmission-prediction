from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ----- Pydantic Model -----
class Patient(BaseModel):
    age: str
    time_in_hospital: int
    num_lab_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int

# Modeli yükle
model = joblib.load("models/readmission_xgb.pkl")

@app.get("/")
def read_root():
    return {"message": "Hospital Readmission Prediction API çalışıyor!"}

@app.post("/predict")
def predict(data: Patient):

    # JSON → dict → DataFrame
    df = pd.DataFrame([data.dict()])

    # Modelin eğitimde gördüğü kolonları al
    expected_cols = model.named_steps["preprocess"].feature_names_in_

    # Eksik kolonları tamamla
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0   # sayısal kolonlar için 0 koyuyoruz

    # Kolon sırasını eşitle
    df = df[expected_cols]

    # Tahmin
    proba = model.predict_proba(df)[0][1]

    return {"risk_score": float(proba)}

