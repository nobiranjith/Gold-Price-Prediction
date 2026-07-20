
# ===================================
# GOLD PRICE PREDICTION USING LSTM
# ===================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1️⃣ Download Gold Data
# -----------------------------
gold = yf.download("GC=F", start="2000-01-01")

if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.droplevel(1)

gold = gold[['Close']]
gold.dropna(inplace=True)

print("Data Loaded Successfully")

# -----------------------------
# 2️⃣ Scale Data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(gold)

# -----------------------------
# 3️⃣ Create Time Sequences
# -----------------------------
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Train-Test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 4️⃣ Build LSTM Model
# -----------------------------
model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Training Started...")
model.fit(X_train, y_train, batch_size=32, epochs=20)

# -----------------------------
# 5️⃣ Predictions
# -----------------------------
predictions = model.predict(X_test)

# Inverse Scaling
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# -----------------------------
# 6️⃣ Evaluation
# -----------------------------
mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)

print("\nModel Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

# -----------------------------
# 7️⃣ Plot Results
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("LSTM Gold Price Prediction")
plt.xlabel("Time")
plt.ylabel("Gold Price (USD)")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 8️⃣ Predict Next Day Using Last 60 Days
# -----------------------------

# Get last 60 days from scaled data
last_60_days = scaled_data[-60:]

# Reshape for LSTM
last_60_days = last_60_days.reshape(1, 60, 1)

# Predict next day
next_day_scaled = model.predict(last_60_days)

# Convert back to original price
next_day_price = scaler.inverse_transform(next_day_scaled)

print("\nNext Day Gold Price Prediction:")
print(next_day_price[0][0])


# -----------------------------
# 9️⃣ Predict Next 30 Days (Future Forecast)
# -----------------------------

future_predictions = []

# Copy last 60 days
current_input = scaled_data[-60:]

for i in range(30):
    
    # Reshape
    current_input_reshaped = current_input.reshape(1, 60, 1)
    
    # Predict next value
    next_prediction = model.predict(current_input_reshaped)
    
    # Store prediction
    future_predictions.append(next_prediction[0][0])
    
    # Slide window (remove first, append new prediction)
    current_input = np.append(current_input[1:], next_prediction)

# Convert predictions back to original prices
future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

print("\nNext 30 Days Forecast:")
print(future_predictions)

# -----------------------------
# 🔟 Plot Future Forecast with Real Dates
# -----------------------------

# Get last date from dataset
last_date = gold.index[-1]

# Create future dates (30 days ahead)
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=30)

# Plot actual recent data + future forecast
plt.figure(figsize=(12,6))

# Plot last 200 real days for better visualization
plt.plot(gold.index[-200:], 
         gold['Close'].values[-200:], 
         label="Recent Actual Prices")

# Plot future predictions with real dates
plt.plot(future_dates, 
         future_predictions, 
         label="30-Day Future Forecast", 
         color='red')

plt.title("Gold Price Forecast Until 2026")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.legend()
plt.grid()
plt.show()
