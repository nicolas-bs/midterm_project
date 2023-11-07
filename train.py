import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score


# Load the dataset
df = pd.read_csv(r'hospital_readmissions.csv')

df['readmitted'] = df['readmitted'].apply(lambda x: x == 'yes')

for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype('category')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Reset the indices of the dataframes
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate the target variable
y_train = df_train['readmitted'].values
y_val = df_val['readmitted'].values
y_test = df_test['readmitted'].values

# Drop the target variable from the data
df_train = df_train.drop(['readmitted'], axis=1)
df_val = df_val.drop(['readmitted'], axis=1)
df_test = df_test.drop(['readmitted'], axis=1)

categorical_cols = ['glucose_test', 'A1Ctest']
ordinal_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetes_med']

# Create a column transformer for preprocessing
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    (OrdinalEncoder(), ordinal_cols),
    remainder='passthrough'
)

X_train = preprocessor.fit_transform(df_train)
X_val = preprocessor.transform(df_val)

feature_names = preprocessor.get_feature_names_out()
feature_names = [name.replace('[', '_').replace(']', '_').replace('<', '_') for name in feature_names]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

param_grid = {
    'eta': [0.1, 0.01, 0.001],  # Learning rate
    'max_depth': [3, 4, 5],    # Maximum tree depth
    'min_child_weight': [10, 15, 20],
    'subsample': [0.7, 0.8, 0.9, 1.0],
     'gamma': [0, 0.1, 0.2, 0.3] # Minimum sum of instance weight (hessian) needed in a child
}

# Create the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    seed=1,
    verbosity=2
)

# Create the grid search object
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',  # Use ROC AUC as the evaluation metric
    cv=3,
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parametes
params = grid_search.best_params_
params['eval_metric'] = 'auc'
print(f'The best params are: {params}')

# Train the XGBoost model and find the best iteration
evals_result = {}
watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(params, dtrain, evals=watchlist, num_boost_round=300, evals_result=evals_result, verbose_eval=20, early_stopping_rounds=20)

best_num_boost_round = model.best_iteration

# Plot the AUC values during training
train_auc = evals_result['train']['auc']
val_auc = evals_result['val']['auc']
best_iteration = model.best_iteration

# Predict using the trained model on the validation set
y_pred = model.predict(dval)
print(roc_auc_score(y_val, y_pred).round(4))
     
# Define a train function for later use
def train(X, y, params, num_boost_round, feature_names):
    X_transformed = preprocessor.transform(X)
    dtrain = xgb.DMatrix(X_transformed, label=y, feature_names=feature_names)
    model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)
    return model

# Define a predict function for later use
def predict(X, preprocessor, model, feature_names):
    X_transformed = preprocessor.transform(X)
    X_Dmat = xgb.DMatrix(X_transformed, feature_names=feature_names)
    y_pred = model.predict(X_Dmat)
    return y_pred

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = []

# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    y_train = df_train['readmitted'].values
    y_val = df_val['readmitted'].values
    model = train(df_train, y_train, params, best_num_boost_round, feature_names)
    y_pred = predict(df_val, preprocessor, model, feature_names)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'AUC on fold {fold+1} is {auc}')

print('Validation results:')
print(f'Mean AUC: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# Train the final model on the full training data
model = train(df_full_train, df_full_train['readmitted'].values, params, best_num_boost_round, feature_names)

# Predict on the test set using the final model
y_pred = predict(df_test, preprocessor, model, feature_names)
auc = roc_auc_score(y_test, y_pred)
print(f'Final model AUC on the test set: {auc:.3f}')

# Save the preprocessor and model to a binary file
output_file = 'xgb_eta01.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((preprocessor, model), f_out)

print(f'The model is saved to {output_file}')
