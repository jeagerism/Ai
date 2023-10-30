import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data from local path
local_data = pd.read_csv(r'C:\Users\oo4dx\Downloads\parkinsons (1)\parkinsons.data')  # Adjust the file path accordingly

# Extract features and labels
features = local_data.loc[:, local_data.columns != 'status'].values[:, 1:] # values use for array format
labels = local_data.loc[:, 'status'].values

# check count input 0 1 
print(local_data['status'].value_counts())

#check labels = input
print(labels)
#  Initialize MinMax Scaler classs for -1 to 1
scaler = MinMaxScaler((-1.00, 1.00))

# fit_transform() method fits to the data and then transforms it.
X = scaler.fit_transform(features)
y = labels

# split the dataset into training and testing sets with 20% of testings
x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.20)

# print("\nx train is \n", x_train)
# print("\nx test is \n", x_test)
# print("\ny train is \n", y_train)
# print("\ny test is \n", y_test)

# Fit the XGBoost model with the training data
model = XGBClassifier()
model.fit(x_train, y_train)

# Predict using the test data
y_pred = model.predict(x_test)

# Print the predicted values
print('\n y_pred \n',y_pred)

# Calculate the predicted probabilities
y_pred_proba = model.predict_proba(x_test)
y_pred_proba_class1 = [pred[1] for pred in y_pred_proba]

# Print the predicted probabilities and actual values
# print('\n y_pred_proba_class1 \n', y_pred_proba_class1)
# for i in range(len(y_pred)):
#     print(f"Data {i+1}: Predicted = {y_pred[i]}, Predicted Probability = {y_pred_proba_class1[i]:.2f}, Actual = {y_test[i]}")

y_prediction = model.predict(x_test)

print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)

data = {'Data': [i+1 for i in range(len(y_pred))],
        'Predicted': y_pred,
        'Predicted Probability': [f"{prob:.4f}" for prob in y_pred_proba_class1],
        'Actual': y_test}

df = pd.DataFrame(data)
print(df.to_string(index=False))