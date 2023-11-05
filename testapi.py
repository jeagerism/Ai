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

# Define the function to predict Parkinson
def predict_parkinson(data):
    # Extract features and labels
    features = local_data.loc[:, ~local_data.columns.isin(['status', 'name'])].values
    labels = local_data.loc[:, 'status'].values

    #  Initialize MinMax Scaler classs for -1 to 1
    scaler = MinMaxScaler((-1.00, 1.00))

    # fit_transform() method fits to the data and then transforms it.
    X = scaler.fit_transform(features)
    y = labels

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the XGBoost model with the training data
    model = XGBClassifier()
    model.fit(x_train, y_train)

    # Calculate the predicted probabilities
    y_pred_proba = model.predict_proba(x_test)
    y_pred_proba_class1 = [pred[1] for pred in y_pred_proba]

    # Return the predicted probability
    return y_pred_proba_class1[0]

# Test the function
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
prediction = predict_parkinson(data)
print(prediction)


app = FastAPI()

@app.post("/predict")
async def predict(data: List[int]) -> Dict:
    prediction = predict_parkinson(data)
    return {"prediction": prediction}

if __name__ == "__testapi__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
