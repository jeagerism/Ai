from typing import Union
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI
from typing import List, Dict
import uvicorn

# Load the data from local path
local_data = pd.read_csv(r'C:\Users\oo4dx\Downloads\parkinsons (1)\parkinsons.data')  # Adjust the file path accordingly

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(data: List[float]) -> float:
    # Define the function to predict Parkinson
    def predict_parkinson(data):
        # Extract features and labels
        features = local_data.loc[:, ~local_data.columns.isin(['status', 'name'])].values
        labels = local_data.loc[:, 'status'].values

        #  Initialize MinMax Scaler class for -1 to 1
        scaler = MinMaxScaler((-1.00, 1.00))

        # fit_transform() method fits to the data and then transforms it.
        X = scaler.fit_transform(features)
        y = labels

        # Fit the XGBoost model with the entire dataset
        model = XGBClassifier()
        model.fit(X, y)

        # Convert the input data into a 2D array for prediction
        input_data = pd.DataFrame([data]).values
        input_data_scaled = scaler.transform(input_data)

        # Calculate the predicted probabilities
        y_pred_proba = model.predict_proba(input_data_scaled)
        y_pred_proba_class1 = y_pred_proba[0][1]  # Probability of the positive class

        return y_pred_proba_class1.item()

    return predict_parkinson(data)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
