# 📁 Enhanced Customer Churn Prediction with GUI + Styling + Database

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Create synthetic dataset
np.random.seed(42)
data = {
    'Tenure': np.random.randint(1, 60, 1000),
    'Cashback_amount': np.random.uniform(50, 1000, 1000),
    'City_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], 1000),
    'Warehouse_to_home': np.random.randint(1, 15, 1000),
    'Order_amount_hike_from_last_year': np.random.uniform(0, 100, 1000),
    'Days_since_last_order': np.random.randint(0, 365, 1000),
    'Satisfaction_score': np.random.randint(1, 6, 1000),
    'Number_of_addresses': np.random.randint(1, 4, 1000),
    'Number_of_devices_registered': np.random.randint(1, 6, 1000),
    'Complain': np.random.choice([0, 1], 1000),
    'Order_count': np.random.randint(1, 50, 1000),
    'hourspendonapp': np.random.uniform(1, 10, 1000),
    'Marital_status': np.random.choice(['Married', 'Single'], 1000),
    'Coupon_used': np.random.choice([0, 1], 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Churn': np.random.choice([0, 1], 1000)
}
df = pd.DataFrame(data)

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_marital = LabelEncoder()
df['Marital_status'] = le_marital.fit_transform(df['Marital_status'])
le_city = LabelEncoder()
df['City_tier'] = le_city.fit_transform(df['City_tier'])

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# SQLite setup
def setup_database():
    conn = sqlite3.connect("churn_predictions.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Tenure INTEGER,
        Cashback_amount REAL,
        City_tier TEXT,
        Warehouse_to_home INTEGER,
        Order_amount_hike REAL,
        Days_since_last_order INTEGER,
        Satisfaction_score REAL,
        Number_of_addresses INTEGER,
        Number_of_devices_registered INTEGER,
        Complain INTEGER,
        Order_count INTEGER,
        hourspendonapp REAL,
        Marital_status TEXT,
        Coupon_used INTEGER,
        Gender TEXT,
        Prediction TEXT
    )''')
    conn.commit()
    conn.close()

setup_database()

# GUI
def predict_churn():
    try:
        inputs = [
            float(entry_tenure.get()),
            float(entry_cashback.get()),
            le_city.transform([city_var.get()])[0],
            float(entry_warehouse.get()),
            float(entry_hike.get()),
            float(entry_days.get()),
            float(entry_score.get()),
            float(entry_address.get()),
            float(entry_devices.get()),
            int(complain_var.get()),
            float(entry_orders.get()),
            float(entry_hours.get()),
            le_marital.transform([marital_var.get()])[0],
            int(coupon_var.get()),
            le_gender.transform([gender_var.get()])[0]
        ]

        scaled_input = scaler.transform([inputs])
        prediction = model.predict(scaled_input)[0]
        result = "Likely to Churn" if prediction == 1 else "Will Stay"

        conn = sqlite3.connect("churn_predictions.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       tuple(list(inputs[:2]) + [city_var.get()] + inputs[3:12] + [marital_var.get(), coupon_var.get(), gender_var.get(), result]))
        conn.commit()
        conn.close()

        messagebox.showinfo("Prediction", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))

app = tk.Tk()
app.title("Customer Churn Predictor")
app.configure(bg="white")
app.geometry("600x600")

# Entry Fields
fields = [
    ("Tenure", "entry_tenure"),
    ("Cashback Amount", "entry_cashback"),
    ("Warehouse to Home", "entry_warehouse"),
    ("Order Amount Hike", "entry_hike"),
    ("Days Since Last Order", "entry_days"),
    ("Satisfaction Score", "entry_score"),
    ("Number of Addresses", "entry_address"),
    ("Number of Devices", "entry_devices"),
    ("Order Count", "entry_orders"),
    ("Hours Spent on App", "entry_hours")
]

entries = {}
row = 0
for i, (label, varname) in enumerate(fields):
    tk.Label(app, text=label, bg="white", fg="black").grid(row=row, column=i % 2 * 2, padx=10, pady=5, sticky='w')
    entry = tk.Entry(app)
    entry.grid(row=row, column=i % 2 * 2 + 1)
    entries[varname] = entry
    if i % 2 != 0:
        row += 1

city_var = tk.StringVar()
marital_var = tk.StringVar()
gender_var = tk.StringVar()
complain_var = tk.IntVar()
coupon_var = tk.IntVar()

tk.Label(app, text="City Tier", bg="white").grid(row=row, column=0, padx=10, pady=5, sticky='w')
tk.OptionMenu(app, city_var, 'Tier 1', 'Tier 2', 'Tier 3').grid(row=row, column=1)
city_var.set('Tier 1')
row += 1

tk.Label(app, text="Marital Status", bg="white").grid(row=row, column=0, padx=10, pady=5, sticky='w')
tk.OptionMenu(app, marital_var, 'Married', 'Single').grid(row=row, column=1)
marital_var.set('Single')
row += 1

tk.Label(app, text="Gender", bg="white").grid(row=row, column=0, padx=10, pady=5, sticky='w')
tk.OptionMenu(app, gender_var, 'Male', 'Female').grid(row=row, column=1)
gender_var.set('Male')
row += 1

tk.Checkbutton(app, text="Complain", variable=complain_var, bg="white").grid(row=row, column=0, sticky='w')
tk.Checkbutton(app, text="Coupon Used", variable=coupon_var, bg="white").grid(row=row, column=1, sticky='w')
row += 1

entry_tenure = entries["entry_tenure"]
entry_cashback = entries["entry_cashback"]
entry_warehouse = entries["entry_warehouse"]
entry_hike = entries["entry_hike"]
entry_days = entries["entry_days"]
entry_score = entries["entry_score"]
entry_address = entries["entry_address"]
entry_devices = entries["entry_devices"]
entry_orders = entries["entry_orders"]
entry_hours = entries["entry_hours"]

tk.Button(app, text="Predict Churn", command=predict_churn, bg='blue', fg='white', font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=20)

app.mainloop()
