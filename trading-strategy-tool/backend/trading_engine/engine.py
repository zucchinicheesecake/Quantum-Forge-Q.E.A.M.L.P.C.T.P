from datetime import datetime
from .strategies.base_strategy import BaseStrategy

class TradingEngine:
    def __init__(self):
        self.strategies = []
        self.current_time = datetime.now()
        
    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)
        
    def run(self):
        for strategy in self.strategies:
            strategy.execute(self.current_time)
