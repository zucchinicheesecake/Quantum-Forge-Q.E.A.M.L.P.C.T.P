import unittest
import numpy as np
from ml_models.strategy_optimizer import StrategyOptimizer

class TestMLModels(unittest.TestCase):
    def setUp(self):
        self.optimizer = StrategyOptimizer()
        self.X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        self.y = [0, 1, 0, 1, 0]
        
    def test_training(self):
        result = self.optimizer.train(self.X, self.y)
        self.assertIsNotNone(result)
        self.assertIn('accuracy', result)
        self.assertIn('feature_importances', result)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
        
    def test_prediction(self):
        self.optimizer.train(self.X, self.y)
        prediction = self.optimizer.predict([3, 4])
        self.assertIn(prediction[0], [0, 1])
        
    def test_invalid_input(self):
        with self.assertRaises(Exception):
            self.optimizer.predict("invalid input")
            
    def test_empty_training(self):
        result = self.optimizer.train([], [])
        self.assertIsNone(result)
