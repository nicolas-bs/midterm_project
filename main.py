import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import xgboost as xgb
from fastapi.responses import JSONResponse
import io  # Add this import

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

@app.post('/Individual')
async def predict(data: InputData):
    input_data = [data.model_dump()]
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

@app.post('/Group')
async def upload_and_predict(file: UploadFile):
    if file.filename.endswith('.csv'):
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('[', '_').replace(']', '_').replace('<', '_') for name in feature_names]
        X_Dm = xgb.DMatrix(X, feature_names=feature_names)
        y_pred = model.predict(X_Dm)
        readmitted = y_pred >= 0.5

        # Convert float32 values to float
        y_pred = [float(val) for val in y_pred]

        # Convert bool values to int (1 for True, 0 for False)
        readmitted = [int(val) for val in readmitted]

        # Calculate the proportion of positive values
        proportion_positive = sum(readmitted) / len(readmitted)

        # Create a list to store the mixed result
        mixed_result = []
        for prob, is_readmitted in zip(y_pred, readmitted):
            mixed_result.append({'probability': prob, 'is_readmitted': is_readmitted})

        # Return the mixed result followed by the proportion
        result = {
            'mixed_result': mixed_result,
            'proportion_positive': proportion_positive
        }
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=400, detail="File must be a CSV")

