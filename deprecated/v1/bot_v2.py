import argparse
import time
import logging
import ccxt
import numpy as np
import pandas as pd
import websocket
import json
import threading
from typing import Any, Dict, List
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(closes: List[float], periods: int = 14) -> float:
        delta = pd.Series(closes).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    @staticmethod
    def calculate_macd(closes: List[float]) -> Dict[str, float]:
        prices = pd.Series(closes)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {"macd": macd.iloc[-1], "signal": signal.iloc[-1]}

    @staticmethod
    def calculate_bollinger_bands(closes: List[float], window: int = 20) -> Dict[str, float]:
        prices = pd.Series(closes)
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return {
            "upper": upper_band.iloc[-1],
            "middle": sma.iloc[-1],
            "lower": lower_band.iloc[-1]
        }

class QuantumTrader:
    def __init__(self):
        self.name = "QuantumTrader"
        self.exchange = ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET_KEY'
        })
        self.indicators = TechnicalIndicators()
        self.ws = None
        self.last_price = None
        self.start_websocket()

    def start_websocket(self):
        def on_message(ws, message):
            data = json.loads(message)
            if 'data' in data and 'price' in data['data']:
                self.last_price = float(data['data']['price'])

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws):
            logger.info("WebSocket connection closed")

        def on_open(ws):
            subscribe_message = {
                "type": "subscribe",
                "topic": "trade.BTCUSD"
            }
            ws.send(json.dumps(subscribe_message))

        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://phemex.com/ws",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def analyze_market(self, symbol: str = 'BTC/USD') -> Dict[str, Any]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
            closes = [x[4] for x in ohlcv]
            
            rsi = self.indicators.calculate_rsi(closes)
            macd = self.indicators.calculate_macd(closes)
            bb = self.indicators.calculate_bollinger_bands(closes)
            
            # Trading strategy
            signal = "NEUTRAL"
            confidence = 0.0
            
            if rsi < 30 and self.last_price < bb['lower']:
                signal = "BUY"
                confidence = 0.8
            elif rsi > 70 and self.last_price > bb['upper']:
                signal = "SELL"
                confidence = 0.8
            elif macd['macd'] > macd['signal']:
                signal = "BUY"
                confidence = 0.6
            elif macd['macd'] < macd['signal']:
                signal = "SELL"
                confidence = 0.6

            return {
                "signal": signal,
                "confidence": confidence,
                "price": self.last_price or closes[-1],
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger_bands": bb
                }
            }
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {"signal": "NEUTRAL", "confidence": 0.0}

class GhostWriter:
    def __init__(self):
        self.name = "GhostWriter"
        self.exchange = ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET_KEY'
        })
        self.positions = {}
        
    def execute_trade(self, signal: Dict[str, Any], position_size: float) -> bool:
        try:
            symbol = 'BTC/USD'
            if signal['signal'] == "BUY":
                order = self.exchange.create_market_buy_order(symbol, position_size)
                stop_price = signal['price'] * 0.98  # 2% stop loss
                self.positions[order['id']] = {
                    'entry_price': signal['price'],
                    'stop_loss': stop_price,
                    'size': position_size
                }
            elif signal['signal'] == "SELL":
                order = self.exchange.create_market_sell_order(symbol, position_size)
                stop_price = signal['price'] * 1.02  # 2% stop loss
                self.positions[order['id']] = {
                    'entry_price': signal['price'],
                    'stop_loss': stop_price,
                    'size': position_size
                }
            return True
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return False

    def check_stop_losses(self, current_price: float):
        for order_id, position in list(self.positions.items()):
            if (position['entry_price'] > current_price and current_price <= position['stop_loss']) or \
               (position['entry_price'] < current_price and current_price >= position['stop_loss']):
                try:
                    self.exchange.create_market_sell_order('BTC/USD', position['size'])
                    del self.positions[order_id]
                    logger.info(f"Stop loss triggered for order {order_id}")
                except Exception as e:
                    logger.error(f"Stop loss execution error: {str(e)}")

class KellyCriterionOptimizer:
    def __init__(self):
        self.name = "KellyCriterionOptimizer"
        self.trade_history = []
        
    def optimize_position_size(self, risk: float, confidence: float) -> float:
        win_rate = self.calculate_win_rate()
        kelly_fraction = win_rate - ((1 - win_rate) / (risk / 0.02))  # Using 2% as base risk
        position_size = max(0.0, min(risk * kelly_fraction, risk))
        return float(Decimal(str(position_size)).quantize(Decimal('0.0001')))
    
    def calculate_win_rate(self) -> float:
        if not self.trade_history:
            return 0.5
        wins = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        return wins / len(self.trade_history)

    def update_trade_history(self, trade_result: Dict):
        self.trade_history.append(trade_result)
        if len(self.trade_history) > 100:  # Keep only last 100 trades
            self.trade_history.pop(0)

def main():
    parser = argparse.ArgumentParser(description='Phemex Trading Bot')
    parser.add_argument('--simulate', action='store_true', help='Run simulation mode')
    parser.add_argument('--live', action='store_true', help='Run live trading mode')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade')
    parser.add_argument('--aggression', type=int, default=5, help='Aggression level for trading')

    args = parser.parse_args()

    try:
        trader = QuantumTrader()
        executor = GhostWriter()
        risk_optimizer = KellyCriterionOptimizer()

        if args.simulate:
            logger.info("Running simulation...")
            run_simulation(trader, executor, risk_optimizer, args.risk, args.aggression)
        elif args.live:
            logger.info("Starting live trading...")
            start_live_trading(trader, executor, risk_optimizer, args.risk, args.aggression)
        else:
            logger.error("Please specify either --simulate or --live mode")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def run_simulation(trader, executor, risk_optimizer, risk: float, aggression: int):
    while True:
        try:
            market_data = trader.analyze_market()
            position_size = risk_optimizer.optimize_position_size(risk, market_data['confidence'])
            
            if market_data['signal'] != "NEUTRAL":
                success = executor.execute_trade(market_data, position_size)
                logger.info(f"Simulation trade: {market_data['signal']} - Success: {success}")
                
                if success:
                    risk_optimizer.update_trade_history({
                        'signal': market_data['signal'],
                        'profit': 0.01  # Simulated profit
                    })
                    
            executor.check_stop_losses(market_data['price'])
            time.sleep(60)
        except Exception as e:
            logger.error(f"Simulation error: {str(e)}")
            break

def start_live_trading(trader, executor, risk_optimizer, risk: float, aggression: int):
    while True:
        try:
            market_data = trader.analyze_market()
            position_size = risk_optimizer.optimize_position_size(risk, market_data['confidence'])
            
            if market_data['signal'] != "NEUTRAL":
                success = executor.execute_trade(market_data, position_size)
                logger.info(f"Live trade: {market_data['signal']} - Success: {success}")
                
                if success:
                    risk_optimizer.update_trade_history({
                        'signal': market_data['signal'],
                        'profit': 0.01  # Actual profit should be calculated
                    })
                    
            executor.check_stop_losses(market_data['price'])
            time.sleep(60)
        except Exception as e:
            logger.error(f"Live trading error: {str(e)}")
            break

if __name__ == '__main__':
    main()
