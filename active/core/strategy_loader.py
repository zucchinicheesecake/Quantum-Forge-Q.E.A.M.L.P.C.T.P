import importlib
import logging
from pathlib import Path
from typing import List, Any

class StrategyLoader:
    def __init__(self):
        self.strategies: List[Any] = []
        self.logger = logging.getLogger(__name__)
        self.load_strategies()
        
    def load_strategies(self):
        """Load all available strategies"""
        strategy_dir = Path(__file__).parent.parent / 'strategies'
        if not strategy_dir.exists():
            self.logger.warning(f"Strategy directory not found: {strategy_dir}")
            return
            
        for strategy_file in strategy_dir.glob('*.py'):
            if strategy_file.name.startswith('_'):
                continue
                
            try:
                module_name = f"strategies.{strategy_file.stem}"
                module = importlib.import_module(module_name)
                if hasattr(module, 'Strategy'):
                    strategy_instance = module.Strategy()
                    self.strategies.append(strategy_instance)
                    self.logger.info(f"Loaded strategy: {strategy_file.stem}")
            except Exception as e:
                self.logger.error(f"Failed to load strategy {strategy_file.stem}: {str}")
                
    def get_active_strategies(self) -> List[Any]:
        """Get all active strategies"""
        return [s for s in self.strategies if s.is_active()]
