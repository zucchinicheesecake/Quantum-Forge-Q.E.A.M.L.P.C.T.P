from typing import Dict, Type, List
import importlib
import os
import logging
from datetime import datetime
from .base_strategy import BaseStrategy

class StrategyLoader:
    def __init__(self, strategies_dir: str = "strategies"):
        self.strategies_dir = strategies_dir
        self.available_strategies: Dict[str, Type[BaseStrategy]] = {}
        self.logger = logging.getLogger(__name__)
        self.strategy_metrics = {}

    def load_strategies(self) -> None:
        """Load all strategy modules from the strategies directory"""
        try:
            strategy_files = [f for f in os.listdir(self.strategies_dir) 
                            if f.endswith('.py') and not f.startswith('__')]
            
            for strategy_file in strategy_files:
                module_name = strategy_file[:-3]
                module_path = f"{self.strategies_dir}.{module_name}"
                
                try:
                    module = importlib.import_module(module_path)
                    strategy_class = getattr(module, module_name)
                    
                    if self.validate_strategy(strategy_class):
                        self.available_strategies[module_name] = strategy_class
                        self.strategy_metrics[module_name] = {
                            'loaded_at': datetime.now(),
                            'execution_count': 0,
                            'success_rate': 0.0
                        }
                        self.logger.info(f"Successfully loaded strategy: {module_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load strategy {module_name}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading strategies: {str(e)}")

    def get_strategy(self, strategy_name: str) -> Type[BaseStrategy]:
        """Get a specific strategy by name"""
        strategy = self.available_strategies.get(strategy_name)
        if strategy:
            self._update_strategy_metrics(strategy_name)
        return strategy

    def list_strategies(self) -> List[str]:
        """Return list of available strategy names"""
        return list(self.available_strategies.keys())

    def validate_strategy(self, strategy_class: Type) -> bool:
        """Validate if a strategy class implements required methods"""
        required_methods = ['initialize', 'analyze', 'execute', 'validate_parameters']
        has_methods = all(hasattr(strategy_class, method) for method in required_methods)
        
        if not has_methods:
            self.logger.warning(f"Strategy validation failed: missing required methods")
            return False
            
        try:
            # Validate strategy inherits from BaseStrategy
            if not issubclass(strategy_class, BaseStrategy):
                self.logger.warning(f"Strategy must inherit from BaseStrategy")
                return False
        except TypeError:
            return False
            
        return True

    def _update_strategy_metrics(self, strategy_name: str) -> None:
        """Update usage metrics for strategy"""
        if strategy_name in self.strategy_metrics:
            self.strategy_metrics[strategy_name]['execution_count'] += 1
            self.strategy_metrics[strategy_name]['last_used'] = datetime.now()

    def get_strategy_metrics(self, strategy_name: str = None) -> Dict:
        """Get metrics for specific strategy or all strategies"""
        if strategy_name:
            return self.strategy_metrics.get(strategy_name, {})
        return self.strategy_metrics
