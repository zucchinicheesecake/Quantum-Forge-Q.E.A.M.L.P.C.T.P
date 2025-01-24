import streamlit as st
import numpy as np
import uuid
import time
from datetime import date

class VirtualExchange:
    def __init__(self):
        self.balance = st.session_state.get('virtual_balance', 100_000.0)
        self.positions = st.session_state.get('virtual_positions', {})
        self.trade_history = st.session_state.get('virtual_trades', [])

    def simulate_order(self, order: dict):
        # Realistic slippage model
        slippage = np.random.normal(loc=0.0003, scale=0.0001)
        filled_price = order['price'] * (1 + slippage)
        
        # Simulate exchange latency
        # Removed for performance
        # time.sleep(0.2)
        
        # Update session state
        st.session_state.virtual_balance = self.balance
        st.session_state.virtual_positions = self.positions
        st.session_state.virtual_trades = self.trade_history
        
        # Check for SL/TP conditions
        self.check_sl_tp(order, filled_price)

        # Log the operation with UUID and timestamp
        self.log_operation(order)

        return {
            "orderID": f"VIRT_{uuid.uuid4().hex[:6]}",
            "avgPrice": filled_price,
            "cumQty": order['orderQty']
        }

    def risk_checks(self, order):
        # Position size limit (5% of balance)
        max_position = self.balance * 0.05 / order['price']
        if order['orderQty'] > max_position:
            raise ValueError(f"Order size exceeds 5% risk limit")
        
        # Daily trade limit
        today_trades = [t for t in self.trade_history if t['date'] == date.today()]
        if len(today_trades) > 100:
            raise ValueError("Daily virtual trade limit reached")
