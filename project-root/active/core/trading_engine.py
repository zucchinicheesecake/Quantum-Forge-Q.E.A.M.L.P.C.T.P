import logging
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
from .strategy_loader import StrategyLoader
from .risk_manager import RiskManager
from .position_manager import PositionManager

class TradingEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_loader = StrategyLoader()
        self.risk_manager = RiskManager(config.get('risk_params', {}))
        self.position_manager = PositionManager()
        self.active_strategies = {}
        self.logger = logging.getLogger(__name__)
        self.market_data = pd.DataFrame()
        self.performance_metrics = {}
        self.trade_history = []

    def initialize(self):
        """Initialize trading engine and load strategies"""
        try:
            self.strategy_loader.load_strategies()
            self._setup_strategies()
            self._initialize_performance_tracking()
            self.logger.info("Trading engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise

    def _setup_strategies(self):
        """Set up selected trading strategies"""
        for strategy_name in self.config['strategies']:
            strategy_class = self.strategy_loader.get_strategy(strategy_name)
            if strategy_class:
                strategy_instance = strategy_class(self.config)
                if strategy_instance.validate_parameters():
                    self.active_strategies[strategy_name] = strategy_instance
                    self.logger.info(f"Strategy {strategy_name} initialized")
                else:
                    self.logger.error(f"Strategy {strategy_name} failed parameter validation")

    def update_market_data(self, new_data: pd.DataFrame):
        """Update market data and trigger analysis"""
        self.market_data = pd.concat([self.market_data, new_data]).tail(1000)
        self._analyze_market()
        self._update_positions()

    def _analyze_market(self):
        """Analyze market data using active strategies"""
        for strategy_name, strategy in self.active_strategies.items():
            try:
                signals = strategy.analyze(self.market_data)
                if self.risk_manager.validate_signals(signals):
                    self._execute_trades(strategy_name, signals)
                else:
                    self.logger.warning(f"Signals rejected by risk manager for {strategy_name}")
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {str(e)}")

    def _execute_trades(self, strategy_name: str, signals: Dict[str, Any]):
        """Execute trading signals"""
        try:
            strategy = self.active_strategies[strategy_name]
            position_size = self.risk_manager.calculate_position_size(signals)
            execution_result = strategy.execute(signals, position_size)
            
            self._record_trade(strategy_name, signals, execution_result)
            self._update_performance_metrics(strategy_name, execution_result)
            
        except Exception as e:
            self.logger.error(f"Trade execution error in {strategy_name}: {str(e)}")

    def _record_trade(self, strategy_name: str, signals: Dict, result: Dict):
        """Record trade details for analysis"""
        trade_record = {
            'timestamp': datetime.now(),
            'strategy': strategy_name,
            'signals': signals,
            'result': result
        }
        self.trade_history.append(trade_record)

    def _update_performance_metrics(self, strategy_name: str, execution_result: Dict[str, Any]):
        """Update performance metrics for strategies"""
        if strategy_name not in self.performance_metrics:
            self.performance_metrics[strategy_name] = []
        
        metrics = {
            'timestamp': datetime.now(),
            'profit_loss': execution_result.get('profit_loss', 0),
            'trade_count': execution_result.get('trade_count', 0),
            'success_rate': execution_result.get('success_rate', 0),
            'risk_metrics': self.risk_manager.get_metrics()
        }
        self.performance_metrics[strategy_name].append(metrics)

    def _update_positions(self):
        """Update and manage open positions"""
        self.position_manager.update_positions(self.market_data)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of trading performance"""
        return {
            'total_trades': len(self.trade_history),
            'active_positions': self.position_manager.get_active_positions(),
            'strategy_metrics': self.strategy_loader.get_strategy_metrics(),
            'risk_metrics': self.risk_manager.get_metrics()
        }
