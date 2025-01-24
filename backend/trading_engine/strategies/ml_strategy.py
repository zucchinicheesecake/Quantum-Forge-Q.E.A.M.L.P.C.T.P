from .base_strategy import BaseStrategy
from ml_models.strategy_optimizer import StrategyOptimizer
import numpy as np

class MLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.optimizer = StrategyOptimizer()
        self.training_data = []
        self.training_labels = []
        
    def add_training_data(self, features, label):
        self.training_data.append(features)
        self.training_labels.append(label)
        
    def train_model(self):
        if len(self.training_data) == 0:
            print("No training data available")
            return False
            
        training_result = self.optimizer.train(
            np.array(self.training_data),
            np.array(self.training_labels)
        )
        
        if training_result:
            print(f"Model trained successfully with accuracy: {training_result['accuracy']}")
            return True
        return False
        
    def execute(self, timestamp):
        if not self.training_data:
            print("Model not trained yet")
            return
            
        market_data = self.get_market_data()
        if not market_data:
            print("No market data available")
            return
            
        prediction = self.optimizer.predict(market_data)
        if prediction == 1:
            self.execute_trade('buy')
        elif prediction == -1:
            self.execute_trade('sell')
