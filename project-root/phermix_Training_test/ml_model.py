import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class MarketPredictor:
    def __init__(self, config):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = config.get('sequence_length', 60)
        self.features = config.get('features', ['close', 'volume', 'rsi', 'macd'])
        
    def prepare_data(self, data: pd.DataFrame):
        scaled_data = self.scaler.fit_transform(data[self.features])
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predicting next close price
            
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame):
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, len(self.features))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        scaled_data = self.scaler.transform(data[self.features].tail(self.sequence_length))
        prediction = self.model.predict(scaled_data.reshape(1, self.sequence_length, len(self.features)))
        return self.scaler.inverse_transform(prediction)[0]
