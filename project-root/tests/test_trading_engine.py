import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from active.core.trading_engine import TradingEngine
from active.core.strategy_loader import StrategyLoader

class TestTradingEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            'strategies': ['TestStrategy'],
            'risk_params': {
                'max_position_size': 1000,
                'max_drawdown': 0.02
            }
        }
        self.engine = TradingEngine(self.config)
        
    def test_initialization(self):
        self.engine.initialize()
        self.assertIsNotNone(self.engine.strategy_loader)
        self.assertIsNotNone(self.engine.risk_manager)
        self.assertIsNotNone(self.engine.position_manager)
        
    def test_market_data_update(self):
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [98, 99, 100],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        self.engine.update_market_data(test_data)
        self.assertFalse(self.engine.market_data.empty)
        self.assertEqual(len(self.engine.market_data), 3)
        
    @patch('active.core.strategy_loader.StrategyLoader.get_strategy')
    def test_strategy_execution(self, mock_get_strategy):
        mock_strategy = Mock()
        mock_strategy.analyze.return_value = {'signal': 'buy', 'price': 100}
        mock_strategy.execute.return_value = {'profit_loss': 10, 'trade_count': 1}
        mock_get_strategy.return_value = mock_strategy
        
        self.engine.initialize()
        test_data = pd.DataFrame({'close': [100]})
        self.engine.update_market_data(test_data)
        
        self.assertTrue(mock_strategy.analyze.called)
        self.assertTrue(mock_strategy.execute.called)
        
    def test_performance_metrics(self):
        self.engine.initialize()
        strategy_name = 'TestStrategy'
        execution_result = {
            'profit_loss': 100,
            'trade_count': 1,
            'success_rate': 1.0
        }
        self.engine._update_performance_metrics(strategy_name, execution_result)
        
        self.assertIn(strategy_name, self.engine.performance_metrics)
        metrics = self.engine.performance_metrics[strategy_name][-1]
        self.assertEqual(metrics['profit_loss'], 100)
        self.assertEqual(metrics['trade_count'], 1)

if __name__ == '__main__':
    unittest.main()
