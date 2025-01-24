import importlib
from pathlib import Path
from typing import List, Any

class StrategyLoader:
    def __init__(self):
        self.strategies = []
        self.load_strategies()
        
    def load_strategies(self):
        strategy_dir = Path(__file__).parent.parent / 'strategies'
        for strategy_file in strategy_dir.glob('*.py'):
            module_name = f"strategies.{strategy_file.stem}"
            module = importlib.import_module(module_name)
            if hasattr(module, 'Strategy'):
                self.strategies.append(module.Strategy())
                
    def get_active_strategies(self) -> List[Any]:
        return [s for s in self.strategies if s.is_active()]
