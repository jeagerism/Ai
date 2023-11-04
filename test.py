import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import simpledialog

# Load the data from local path
local_data = pd.read_csv(r'C:\Users\oo4dx\Downloads\parkinsons (1)\parkinsons.data')  # Adjust the file path accordingly

# Set parameter Header
headers = local_data.columns
filtered_headers = [header for header in headers if header not in ['name', 'status']]
print("ชื่อหัวที่ไม่ใช่ 'name' และ 'status' คือ:")
print(filtered_headers)

# Extract features and labels
features = local_data.loc[:, ~local_data.columns.isin(['status', 'name'])].values # values use for array format
labels = local_data.loc[:, 'status'].values
print('\nFeathers[0] = \n',features[0],type(features[0]))
#  Initialize MinMax Scaler classs for -1 to 1
scaler = MinMaxScaler((-1.00, 1.00))

# fit_transform() method fits to the data and then transforms it.
X = scaler.fit_transform(features)
y = labels

# split the dataset into training and testing sets with 20% of testings
x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.20)

# Fit the XGBoost model with the training data
model = XGBClassifier()
model.fit(x_train, y_train)

# Calculate the predicted probabilities
y_pred_proba = model.predict_proba(x_test)
y_pred_proba_class1 = [pred[1] for pred in y_pred_proba]


# Predict using the test data
y_pred = model.predict(x_test)
y_prediction = model.predict(x_test)

print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)

data = {'Data': [i+1 for i in range(len(y_pred))],
        'Predicted': y_pred,
        'Predicted Probability': [f"{prob:.4f}" for prob in y_pred_proba_class1],
        'Actual': y_test}

df = pd.DataFrame(data)
print(df.to_string(index=False))


# ฟังก์ชันสำหรับคำนวณโอกาส Parkinson's จาก input
def calculate_parkinsons_probability(inputs):
    input_array = [[inputs[i] for i in range(len(inputs))]]
    input_scaled = scaler.transform(input_array)
    return model.predict_proba(input_scaled)

# สร้างหน้าต่างหลักของแอปพลิเคชัน
root = tk.Tk()

# สร้างช่อง input สำหรับผู้ใช้ป้อนข้อมูล
inputs = []
for i in range(len(filtered_headers)):
    user_input = simpledialog.askfloat("Input", f"Enter feature {filtered_headers[i]}:")
    inputs.append(user_input)

# คำนวณโอกาสที่เป็น Parkinson's จาก input ที่ได้รับ
parkinsons_probability = calculate_parkinsons_probability(inputs)

# แสดงผลลัพธ์
result_text = f"The probability of having Parkinson's is approximately: {parkinsons_probability[0][1]*100:.4f}%"
result_label = tk.Label(root, text=result_text)
result_label.pack()
print(f"The probability of having Parkinson's is approximately: {parkinsons_probability[0][1]*100:.4f}%")
# แสดงหน้าต่าง
root.mainloop()
