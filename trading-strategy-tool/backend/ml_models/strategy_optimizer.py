import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class StrategyOptimizer:
    def __init__(self):
        self.model = RandomForestClassifier()
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)
