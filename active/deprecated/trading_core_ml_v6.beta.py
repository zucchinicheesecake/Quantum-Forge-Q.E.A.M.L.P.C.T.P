"""  
Phemex Trading Bot - Machine Learning Core (v6.0-beta)  
Last Modified: 2024-03-15  
"""  

import numpy as np  
import tensorflow as tf  
from typing import Tuple  

# ------------------------------  
# Synthetic Data Generation (v6)  
# ------------------------------  
def generate_synthetic_data_v6(num_samples: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:  
    """  
    Generates synthetic market data for training/testing  
    Features: OHLCV + volatility + funding rate  
    Targets: Next 5-minute price movement (0=down, 1=up)  
    """  
    np.random.seed(42)  
    time_steps = 10  # Number of historical intervals per sample
    features = 6     # OHLCV + volatility
    
    # Generate sufficient data points
    X = np.random.randn(num_samples * time_steps, features)  # Key fix: Multiply samples by time steps
    X = X.reshape(num_samples, time_steps, features)  # Now (10_000, 10, 6)
    
    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standard normalization
    
    y = np.random.randint(0, 2, num_samples)  # Binary labels  
    return X, y  

# ------------------------  
# Neural Network Model (v6)  
# ------------------------  
def train_model(data: Tuple[np.ndarray, np.ndarray]) -> tf.keras.Model:  
    """  
    Trains LSTM-based price predictor  
    Args:  
        data: (X_train, y_train) from generate_synthetic_data_v6()  
    Returns:  
        Trained TF model ready for live prediction  
    """  
    X, y = data  
    model = tf.keras.Sequential([  
        tf.keras.layers.LSTM(64, input_shape=(10, 6)),  
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])  
    model.compile(loss='binary_crossentropy', optimizer='adam')  
    model.fit(X, y, epochs=10, batch_size=32)  
    return model  

# -------------------------  
# Execution Flow (New in v6)  
# -------------------------  
if __name__ == "__main__":  
    # Auto-generate data -> train -> save model  
    synthetic_data = generate_synthetic_data_v6()  
    trained_model = train_model(synthetic_data)  
    trained_model.save("models/lstm_price_predictor_v6.keras")  
    print("✅ Training complete - model saved to /models")
