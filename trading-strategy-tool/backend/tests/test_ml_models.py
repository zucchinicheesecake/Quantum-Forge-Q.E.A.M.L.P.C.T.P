import unittest
from ml_models.strategy_optimizer import StrategyOptimizer

class TestMLModels(unittest.TestCase):
    def test_strategy_optimizer(self):
        optimizer = StrategyOptimizer()
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        accuracy = optimizer.train(X, y)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
