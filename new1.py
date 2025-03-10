from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from pycaret.time_series import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.utils.time_series import clean_time_index
import holidays
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(
    "daily_csv/export-jsps014-1d.csv", parse_dates=["timestamp"], index_col="timestamp"
)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
print("Index เป็น unique หรือไม่:", df.index.is_unique)

df = df[-380:]
# ลบ index ที่ซ้ำกันและตั้งค่าความถี่เป็นรายวัน
df = df[~df.index.duplicated(keep="last")]
df = df.interpolate(method="spline", order=2)
df = df.asfreq("D")

# กำหนดเกณฑ์สำหรับค่า pm_2_5
df = df[(df["pm_2_5"] >= 0) & (df["pm_2_5"] <= 100)]
df = df[(df["pm_2_5_sp"] >= 0) & (df["pm_2_5"] <= 130)]

# ใช้เฉพาะคอลัมน์ที่ต้องการ
df = df[["pm_2_5", "pm_2_5_sp"]]  # ใช้แค่ pm_2_5 สำหรับการทำนาย

for lag in range(31, 39):
    df[f"pm_2_5_sp_lag{lag}"] = df["pm_2_5_sp"].shift(lag)

# สร้าง Lag Features (ค่า PM 2.5 ย้อนหลัง 1-7 วัน)
for lag in range(31, 39):
    df[f"pm_2_5_lag{lag}"] = df["pm_2_5"].shift(lag)

# แบ่งข้อมูลเป็น train set และ test set
train_size = len(df) - 15
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]


# เติมค่าหายไปใน train set
train_df = train_df.interpolate(method="linear", limit_direction="forward")
train_df = train_df.clip(lower=0)  # กำหนดให้ค่าต่ำสุดเป็น 0

test_df = test_df.interpolate(method="linear", limit_direction="forward")
test_df = test_df.clip(lower=0)  # กำหนดให้ค่าต่ำสุดเป็น 0

train_df = train_df.drop

# พล็อตข้อมูลอนุกรมเวลา
plt.figure(figsize=(10, 6))
plt.plot(train_df.index.to_timestamp(), train_df["pm_2_5"], label="Train Data")
plt.title("Daily pm_2_5 Data train_df")
plt.xlabel("Date")
plt.ylabel("pm_2_5")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_df.index.to_timestamp(), test_df["pm_2_5"], label="Test Data")
plt.title("Daily pm_2_5 Data test_df")
plt.xlabel("Date")
plt.ylabel("pm_2_5")
plt.legend()
plt.show()

# ตรวจสอบประเภทข้อมูลของ train_df
# print(train_df.dtypes)


# ฟีเจอร์ที่ใช้ในโมเดล
features = []


# 🔹 **ตอนนี้ test_df มีทุกค่าพร้อมให้ใช้ทำนาย pm_2_5_sp**
# บันทึกผลลัพธ์ใน test_df
test_df[feature] = np.maximum(forecast, 0)  # ตรวจสอบค่าไม่ติดลบ


# เติมข้อมูลใน test_df ด้วย Linear Interpolation
test_df = test_df.interpolate(method="linear", limit_direction="forward")
print(test_df)

# ตั้งค่า PyCaret
exp = TSForecastingExperiment()
exp.setup(
    data=train_df[
        [
            "pm_2_5",
        ]
    ],  # ใช้ train_df ทั้งหมด
    target="pm_2_5",
    session_id=123,
    fh=15,
    use_gpu=True,
)

# สร้างโมเดล ARIMA ด้วย Hyperparameters ที่ปรับแล้ว
model = exp.create_model("arima", order=(5, 1, 1), seasonal_order=(5, 1, 1, 12))

model = exp.finalize_model(model)

# ทำนายค่า pm_2_5
forecast = exp.predict_model(
    model, fh=15, X=test_df[[]]
)  # ใช้ฟีเจอร์ที่ผ่าน Differencing และฟีเจอร์อื่น ๆ

# ปรับค่าทำนายให้ไม่เป็นลบ
forecast["y_pred"] = np.maximum(forecast["y_pred"], 0)

# แสดงผลลัพธ์
print("Forecast:")
print(forecast)

# แก้ไขการพล็อตกราฟเปรียบเทียบค่าจริงกับค่าทำนาย
plt.figure(figsize=(10, 6))
plt.plot(df.index[-50:].to_timestamp(), df["pm_2_5"][-50:], label="Actual", marker="o")
plt.plot(
    forecast.index.to_timestamp(),
    forecast["y_pred"],
    label="Forecast",
    marker="s",
    linestyle="dashed",
)
plt.xlabel("Date")
plt.ylabel("pm_2_5")
plt.title("Actual vs Forecasted pm_2_5")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
