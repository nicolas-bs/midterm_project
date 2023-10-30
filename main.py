import pickle
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
model_file = 'xgb_eta01.bin'

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

# Load the preprocessor and model
with open(model_file, 'rb') as f_in:
    preprocessor, model = pickle.load(f_in)

@app.post('/predict')
async def predict(data: InputData):
    try:
        # Convert the input data to a Pandas DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Transform the input data using the preprocessor
        X = preprocessor.transform(input_data)

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('[', '_').replace(']', '_').replace('<', '_') for name in feature_names]

        # Create a DMatrix
        X_Dm = xgb.DMatrix(X, feature_names=feature_names)

        # Make predictions
        y_pred = model.predict(X_Dm)

        # Check if the prediction is above a certain threshold (e.g., 0.5)
        readmitted = y_pred >= 0.5

        result = {
            'readmitted_probability': float(y_pred[0]),
            'readmitted': bool(readmitted[0])
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5049)
