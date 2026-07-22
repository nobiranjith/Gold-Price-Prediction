Gold Price Prediction Using LSTM Deep Learning
📌 Overview

Gold Price Prediction Using LSTM Deep Learning is an AI-powered financial forecasting project that predicts future gold prices using historical market data and deep learning techniques. The project leverages a Long Short-Term Memory (LSTM) neural network to capture long-term dependencies in time-series data, enabling accurate prediction of gold price trends.

To improve prediction performance, the model can incorporate external economic indicators such as the US Dollar Index (DXY), Crude Oil Prices (WTI), and the S&P 500 Index, which significantly influence the global gold market. This project demonstrates the practical application of Artificial Intelligence and Deep Learning in financial analytics and investment decision support.

🚀 Features
📈 Historical Gold Price Analysis
🤖 LSTM-based Deep Learning Model
📊 Time-Series Forecasting
🌍 External Economic Indicators Integration
📉 Actual vs Predicted Price Visualization
📅 Future Gold Price Forecasting
📊 Model Evaluation using MAE and R² Score
🧹 Automatic Data Preprocessing and Normalization

🛠️ Technologies Used
Python 3.x
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Scikit-learn
Yahoo Finance API (yfinance)

📂 Project Structure
Gold-Price-Prediction/
│
├── main.py                 # Main application
├── lstm_model.py           # LSTM Model
├── arima_model.py          # ARIMA Model (Comparison)
├── gold_data.csv           # Dataset
├── requirements.txt        # Required libraries
├── README.md
└── images/
      ├── prediction.png
      └── forecast.png

      
📊 Dataset

The historical gold price data is collected from Yahoo Finance using the yfinance library.

Ticker Symbol

GC=F

Additional market indicators used in the project include:

Gold Futures (GC=F)
US Dollar Index (DXY)
Crude Oil Futures (CL=F)
S&P 500 Index (^GSPC)
🧠 Model Architecture

The project uses a stacked LSTM neural network consisting of:

Input Layer
LSTM Layer (100 Units)
Dropout Layer (0.2)
LSTM Layer (100 Units)
Dropout Layer (0.2)
Dense Layer (25 Units)
Output Layer (1 Unit)

Loss Function

Mean Squared Error (MSE)

Optimizer

Adam
⚙️ Workflow
Historical Gold Data
          │
          ▼
 Data Collection (Yahoo Finance)
          │
          ▼
 Data Cleaning & Preprocessing
          │
          ▼
 Feature Scaling (MinMaxScaler)
          │
          ▼
 Sequence Generation
          │
          ▼
 LSTM Model Training
          │
          ▼
 Model Evaluation
          │
          ▼
 Gold Price Prediction
          │
          ▼
 Future Forecast
📈 Model Performance

The model is evaluated using:


Mean Absolute Error (MAE)
R² Score
Training Loss
Validation Loss


The prediction graph compares:

Actual Gold Price
Predicted Gold Price

to assess model accuracy visually.


📷 Output

The project generates:

Actual vs Predicted Gold Price Graph
Future Gold Price Forecast
Model Performance Metrics
Training Progress

Example outputs include:

Gold Price Prediction using LSTM
Future Gold Price Forecast
MAE Score
R² Score
▶️ Installation

Clone the repository:

git clone https://github.com/yourusername/Gold-Price-Prediction.git

Navigate to the project folder:

cd Gold-Price-Prediction

Install dependencies:

pip install -r requirements.txt

Run the project:

python main.py
📦 Required Libraries
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install yfinance

Or simply install everything using:

pip install -r requirements.txt


🎯 Future Improvements
Predict 6-month and 1-year gold prices
Integrate real-time economic indicators
Add sentiment analysis from financial news
Develop a Streamlit or Flask web application
Deploy the model on the cloud
Compare LSTM with Transformer and GRU models
Hyperparameter optimization for improved accuracy
📚 Learning Outcomes


This project demonstrates:

Time-Series Forecasting
Deep Learning with LSTM
Financial Market Analysis
Data Preprocessing
Feature Engineering
Model Evaluation
Data Visualization
AI-based Decision Support Systems


👨‍💻 Author

Ranjith A
B.Tech – Artificial Intelligence & Data Science
Passionate about Artificial Intelligence, Deep Learning, Data Science, Machine Learning, and building AI-powered solutions for real-world financial and business problems.

⭐ Support
If you found this project helpful, please consider starring ⭐ the repository and sharing your feedback. Contributions, suggestions, and improvements are always welcome!
