# Y2CPE410

ที่นี่เป็นส่วนของโปรเจกต์ที่เราพัฒนาขึ้นเพื่อทำนาย โรคพาร์กินสัน

## วิธีการติดตั้ง

1. ดาวน์โหลดโปรเจกต์นี้  File ที่ใช้คือ main.py
2. ติดตั้งไลบรารีที่จำเป็นโดยใช้คำสั่ง:

   ```bash
   pip install fastapi uvicorn xgboost pandas sklearn

3.ตัวอย่างข้อมูลที่ส่งไป api
      
       [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285,         -4.813031, 0.266482, 2.301442, 0.284654]

4.URL API Method POST

   ```bash
      http://127.0.0.1:8000/predict

5.ตัวอย่าง ผลลัพธ์
   {
    "prediction": 0.9982302784919739
   }
