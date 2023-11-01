# Hospital Readmissions Prediction with XGBoost-Classifier

This is a README file for a machine learning project that predicts hospital readmissions using XGBoost. The project involves data preprocessing, model training, and model evaluation.

![Hospital Room](https://www.hopkinsmedicine.org/-/media/patient-care/images/patient-rooms-1.jpg)

### Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Data](#data)
- [Training](#training)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Testing](#testing)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)
- [FastAPI Application](#fastapi-application)
- [Files for Python Development and Containerization](#files-for-python-development-and-containerization)
- [Conclusion](#conclusion)
- [Demonstration](#demonstration)

### Overview
In this project, we use the XGBoost Classifier algorithm to build a predictive model for hospital readmissions. We start by loading and preprocessing the dataset, splitting it into training, validation, and test sets, and encoding categorical and ordinal features. We then train an XGBoost model, evaluate its performance, and save the trained model for future use.

### Getting Started
Before you can run this project, make sure you have the required libraries installed. You can install them using the following commands:
```python
 pip install pandas
 pip install xgboost
 pip install scikit-learn
```
### Data
The dataset used in this project is loaded from a CSV file named `hospital_readmissions.csv.` The target variable is readmitted, which indicates whether a patient was readmitted to the hospital. The data preprocessing steps include converting the target variable to a binary format and encoding categorical and ordinal features.

#### Preprocessing

- Categorical columns: glucose_test, A1Ctest
- Ordinal columns: age, medical_specialty, diag_1, diag_2, diag_3, change, diabetes_med

A column transformer is used to apply appropriate encodings to the features. One-hot encoding is used for categorical features, and ordinal encoding is used for ordinal features.

#### Training
We train an XGBoost model with the following hyperparameters:

- Learning rate (eta): 0.1
- Maximum depth of trees: 4
- Minimum child weight: 5
- Objective: Binary logistic
- Random seed: 1
- Gamma: 1

Evaluation metric: AUC (Area Under the Receiver Operating Characteristic curve)
The number of boosting rounds is set to 105. We train the model using the training data and evaluate it on the validation data.

### K-Fold Cross-Validation
To ensure the model's robustness, we perform K-Fold cross-validation with K=10. This helps us assess the model's performance on different subsets of the data. The mean AUC and standard deviation of AUC across the folds are reported to provide a better understanding of model performance.

#### Testing
After cross-validation, we train the final model using the entire training dataset and evaluate it on the test set. The AUC score is reported as the final performance metric.

#### Saving the model
The trained XGBoost model and the preprocessing transformers are saved to a binary file named xgb_eta01.bin using the pickle module. This allows for reusing the model without the need for retraining.

https://github.com/nicolas-bs/midterm_proyect/assets/69317512/6fae4ac5-958b-4a1a-8ca6-e59ba3ec9165

#### Usage
You can use the saved model for making predictions on new data. Here's an example of how to load the model and make predictions:

```python
import pickle
import xgboost as xgb

# Load the saved model
with open('xgb_eta01.bin', 'rb') as f_in:
    preprocessor, model = pickle.load(f_in)

# Your new data (X_new) should be in the same format as the training data
X_new = preprocessor.transform(X_new)
X_new_dmat = xgb.DMatrix(X_new, feature_names=feature_names)
y_pred = model.predict(X_new_dmat)
```

### FastApi Application
The FastAPI application includes two endpoints:

#### 1. Individual Prediction (/Individual)

This endpoint allows you to make predictions for individual patient data. The input data is provided as a JSON request body in the following format:
```JSON
{
    "age": "string",
    "time_in_hospital": int,
    "n_lab_procedures": int,
    "n_procedures": int,
    "n_medications": int,
    "n_outpatient": int,
    "n_inpatient": int,
    "n_emergency": int,
    "medical_specialty": "string",
    "diag_1": "string",
    "diag_2": "string",
    "diag_3": "string",
    "glucose_test": "string",
    "A1Ctest": "string",
    "change": "string",
    "diabetes_med": "string"
}
```
The application processes the input data, pre-processes it, and makes predictions. It returns the readmission probability and a binary readmitted status for the provided patient data.

#### 2. Group Prediction (/Group)
This endpoint allows you to upload a CSV file containing multiple patient records for prediction. The file should have the same structure as the training data. The application processes the uploaded file, pre-processes the data, makes predictions, and returns a mixed result of readmission probabilities and binary readmitted status for each patient record. Additionally, it calculates the proportion of positive readmitted cases in the group data.

### Files for Python Development and Containerization

1) Dockerfile: Contains instructions to set up the container environment, install software, and configure the application. Ensures consistent application deployment in isolated containers.
   ```Dockerfile
   FROM python:3.9-slim

   RUN pip install pipenv
   
   WORKDIR /app
   COPY ["Pipfile", "Pipfile.lock", "./"]
   
   RUN pipenv install --system --deploy
   
   COPY ["main.py", "xgb_eta01.bin", "./"]
   
   EXPOSE 8000
   
   ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2) Pipfile: Lists project dependencies and their versions. Used with tools like Pipenv for managing Python project environments.
3) Pipfile.lock :  Lock file generated by Pipenv, ensuring that the same package versions are installed when recreating a virtual environment. Guarantees reproducible Python environments.

### Conclusion
This project demonstrates how to create a FastAPI application for making hospital readmission predictions using a pre-trained XGBoost model and data preprocessing transformers. The application provides endpoints for both individual and group predictions, making it useful for various scenarios in healthcare analytics.

### Demonstration
   
   https://github.com/nicolas-bs/midterm_proyect/assets/69317512/777031cf-d912-4c3a-b0d8-08858a1b8b44

