import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
import xgboost as xgb
from pydantic import BaseModel

app = FastAPI()

model_file = 'xgb_eta01.bin'

with open(model_file, 'rb') as f_in:
    preprocessor, model = pickle.load(f_in)

class InputData(BaseModel):
    age: str
    time_in_hospital: int
    n_lab_procedures: int
    n_procedures: int
    n_medications: int
    n_outpatient: int
    n_inpatient: int
    n_emergency: int
    medical_specialty: str
    diag_1: str
    diag_2: str
    diag_3: str
    glucose_test: str
    A1Ctest: str
    change: str
    diabetes_med: str

@app.post('/predict')
async def predict(data: InputData):

    input_data = [data.dict()]
    input_df = pd.DataFrame(input_data)

    X = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace('[', '_').replace(']', '_').replace('<', '_') for name in feature_names]

    X_Dm = xgb.DMatrix(X, feature_names=feature_names)
    y_pred = model.predict(X_Dm)

    readmitted = y_pred >= 0.5

    result = {
        'readmitted_probability': float(y_pred[0]),
        'readmitted': bool(readmitted[0])
    }
    return result