#import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import simpledialog

# Load the data from local path
local_data = pd.read_csv(r'C:\Users\oo4dx\Downloads\parkinsons (1)\telemonitoring\__pycache__\parkinsons.data')  # Adjust the file path accordingly

# Set parameter Header
headers = local_data.columns
filtered_headers = [header for header in headers if header not in ['name', 'status']]

#แยก featuers ที่จะใช้ทดสอบโดยเป็น input ไปในการเทรนโมเดล
#แยก labels เพื่อเป็น output ที่จะใช้ในการเทรนโมเดล
features = local_data.loc[:, ~local_data.columns.isin(['status', 'name'])].values # values use for array format
labels = local_data.loc[:, 'status'].values

# กำหนดค่า min max เพื่อปรับข้อมูลให้อยู่ในช่วงที่ต้องการซึ่งสามารถช่วยให้การเทรนโมเดลที่มีประสิทธิภาพมากขึ้นได้ 
scaler = MinMaxScaler((-1.00, 1.00))

# X = scaler.fit_transform(features) จะทำการปรับข้อมูล features ให้อยู่ในช่วงที่กำหนดไว้ ซึ่งจะใช้ในกระบวนการเทรนโมเดล เพื่อให้โมเดลเรียนรู้และทำนายข้อมูลได้อย่างมีประสิทธิภาพ 
# ส่วน y = labels จะเป็นการกำหนดข้อมูล labels ที่ใช้ในกระบวนการเทรนโมเดล
X = scaler.fit_transform(features)
y = labels

# แบ่งdataset ออกไปทำการเทรนโมเดล 80% และ ทดสอบโมเดล 20%
x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.20)

# สร้างตัวแบบ (model) ที่ใช้ในการฝึก (training) และทำนาย (prediction) โดยใช้ XGBoost algorithm ซึ่งเป็นอัลกอริทึมการเรียนรู้เชิงเส้นแบบที่มีประสิทธิภาพสำหรับงานที่มีข้อมูลขนาดใหญ่ หรืองานที่มีการทำนายที่แม่นยำได้.
model = XGBClassifier()
model.fit(x_train, y_train)

# ใช้โมเดลที่เรียนรู้แล้วเพื่อทำนายความน่าจะเป็น (probability) ของข้อมูลทดสอบ (x_test) โดยผลลัพธ์ที่ได้จะเป็นค่าความน่าจะเป็น
y_pred_proba = model.predict_proba(x_test)
y_pred_proba_class1 = [pred[1] for pred in y_pred_proba]


# model.predict(x_test) จะใช้โมเดลที่เทรนแล้วเพื่อทำการทำนายผลลัพธ์สำหรับข้อมูลทดสอบ (x_test)
y_pred = model.predict(x_test)
y_prediction = model.predict(x_test)


#accuracy_score(y_test, y_prediction) ใช้สำหรับคำนวณค่าความแม่นยำของโมเดลที่ทำนายคลาสของข้อมูลทดสอบ (y_test) เทียบกับผลลัพธ์ที่ทำนายได้ (y_prediction) จากนั้นค่านี้จะถูกคูณด้วย 100 เพื่อแสดงเป็นเปอร์เซ็นต์ ทำให้เราสามารถทราบถึงประสิทธิภาพของโมเดลในการทำนายคลาสของข้อมูลทดสอบได้อย่างง่ายดาย
print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)


"""
สร้าง DataFrame ขึ้นมาและนำมาแสดงผลในรูปแบบของตาราง โดยมีคอลัมน์ดังนี้:

'Data': เป็นลำดับของข้อมูลที่ถูกทำนาย
'Predicted': เป็นความน่าจะเป็นพาร์กินสันที่ถูกทำนายจากโมเดล 1 หรือ 0
'Predicted Probability': เป็นเลขความน่าจะเป็นพาร์กินสันมีโอกาสกี่%
'Actual': ค่าความจริงตาม dataset ว่าเป็นพาร์กินสันหรือไม่

โดยจะทำการ loop แสดงค่าทุกๆค่าของ y_pred_proba_class1
"""

data = {'Data': [i+1 for i in range(len(y_pred))],
        'Predicted': y_pred,
        'Predicted Probability': [f"{prob:.4f}" for prob in y_pred_proba_class1],
        'Actual': y_test}

df = pd.DataFrame(data)
print(df.to_string(index=False))


# ฟังก์ชันสำหรับคำนวณโอกาส Parkinson's จาก input
def calculate_parkinsons_probability(inputs):
    #การสร้างอาร์เรย์ input_array ซึ่งจะเป็นอาร์เรย์ที่มีข้อมูลของผู้ใช้ที่ป้อนเข้ามา
    input_array = [[inputs[i] for i in range(len(inputs))]]

    #นำ input_array ที่ได้ไปทำการปรับข้อมูลให้อยู่ในช่วงที่ถูกกำหนดไว้ด้วย MinMaxScaler โดยใช้ฟังก์ชัน scaler.transform
    input_scaled = scaler.transform(input_array)
    return model.predict_proba(input_scaled)

# สร้างหน้าต่างหลักของแอปพลิเคชัน
root = tk.Tk()

# สร้างช่อง input สำหรับผู้ใช้ป้อนข้อมูล
inputs = []

#Loop เพื่อรับinput ตามจำนวน input ที่ใส่ในการเทรน model
for i in range(len(filtered_headers)):
    user_input = simpledialog.askfloat("Input", f"Enter feature {filtered_headers[i]}:")
    inputs.append(user_input)

# เรียกใช้ฟังก์ชันเพื่อคำนวณโอกาสที่เป็น Parkinson's จาก input ที่ได้รับ
parkinsons_probability = calculate_parkinsons_probability(inputs)

# แสดงผลลัพธ์ ใน terminal 
result_text = f"The probability of having Parkinson's is approximately: {parkinsons_probability[0][1]*100:.4f}%"
result_label = tk.Label(root, text=result_text)
result_label.pack()
print(f"The probability of having Parkinson's is approximately: {parkinsons_probability[0][1]*100:.4f}%")

# แสดงหน้าต่าง
root.mainloop()
