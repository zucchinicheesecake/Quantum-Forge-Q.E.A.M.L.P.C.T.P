import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

class StrategyOptimizer:
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        try:
            # Ensure inputs are numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'feature_importances': self.model.steps[1][1].feature_importances_
            }
        except Exception as e:
            print(f"Training error: {str(e)}")
            return None
    
    def predict(self, X):
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            return self.model.predict(X)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
