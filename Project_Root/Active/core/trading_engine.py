from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .strategy_loader import StrategyLoader
from ..adapters.phemex_api import PhemexAPI
from ..Quantum.graph_processor import MarketTopologyAnalyzer

class TradingEngine:
    def __init__(self, config: Dict[str, Any]):
        self.strategy_loader = StrategyLoader()
        self.exchange_adapter = PhemexAPI(config['exchange'])
        self.market_analyzer = MarketTopologyAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def run(self):
        while True:
            market_data = self.exchange_adapter.get_market_data()
            topology = self.market_analyzer.analyze(market_data)
            strategies = self.strategy_loader.get_active_strategies()
            
            for strategy in strategies:
                self.executor.submit(
                    self.execute_strategy,
                    strategy,
                    market_data,
                    topology
                )
                
    def execute_strategy(self, strategy, market_data, topology):
        signal = strategy.generate_signal(market_data, topology)
        if signal:
            self.exchange_adapter.execute_order(signal)
