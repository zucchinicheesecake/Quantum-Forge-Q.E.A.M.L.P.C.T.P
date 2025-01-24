import streamlit as st
from phemex_api import PhemexTradingCore, PhemexAPIError
import asyncio
import pandas as pd
from virtual_exchange import VirtualExchange
from virtual_phemex_exchange import VirtualPhemexExchange  # Import the new class
from datetime import datetime

# Sidebar for Testnet Mode
TESTNET_MODE = st.sidebar.checkbox("🌐 Phemex Testnet", True)

# Initialize the Phemex client with testnet flag
client = PhemexTradingCore(testnet=TESTNET_MODE)

# Initialize the Virtual Exchange for Paper Trading
virtual_exchange = VirtualExchange()

# Initialize the Virtual Phemex Exchange for Testnet Simulation
if TESTNET_MODE:
    virtual_phemex_exchange = VirtualPhemexExchange()

# Sidebar for API Connection
with st.sidebar:
    st.subheader("Phemex Connection")
    if st.button("🔌 Test API Connection"):
        try:
            orderbook = asyncio.run(client.get_orderbook())
            st.success(f"✅ Connected - BTC/USD Bid: {orderbook['book']['bids'][0][0]}")
        except PhemexAPIError as e:
            st.error(f"Connection failed: {str(e)}")

# Display mUSD Balance in Testnet Mode
if TESTNET_MODE:
    st.info(f"Testnet Balance: {client.get_mUSD_balance():.2f} mUSD")
    st.button("🔄 Reset mUSD Balance", help="Reset to 10,000 mUSD")
else:
    st.warning("LIVE TRADING ACTIVE - Real Funds at Risk")

# Debugging output for order book response
st.write("Order Book Response:")
orderbook = asyncio.run(client.get_orderbook())
st.write(orderbook)  # Display the raw response

# Main Dashboard
st.title("Phemex Trading Bot Dashboard")
st.header("Real-Time Market Data")

# Add Paper Trading Toggle
PAPER_MODE = st.sidebar.checkbox(
    "📝 Paper Trading Mode", 
    value=True,
    help="Simulate trades without real capital"
)

if PAPER_MODE:
    st.sidebar.success("Paper Trading Mode is ON")
else:
    st.sidebar.warning("Paper Trading Mode is OFF")

# Backtest Configuration
with st.expander("⚙️ Backtest Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        strategy_type = st.selectbox("Strategy", ["ML-Driven", "Mean Reversion"])
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        initial_balance = st.number_input("Initial Balance", 1000, 1_000_000, 50_000)

# Display Order Book
if 'book' in orderbook:
    st.metric("BTC/USD Bid-Ask", 
              f"{orderbook['book']['bids'][0][0]} | {orderbook['book']['asks'][0][0]}")
else:
    st.error("Order book data is not available.")

# Order Entry Form
with st.expander("📈 Place Manual Order"):
    with st.form("order_form"):
        symbol = st.selectbox("Pair", ["sBTCUSD", "sETHUSD"])  # Use testnet symbols
        side = st.radio("Side", ["Buy", "Sell"])
        quantity = st.number_input("Quantity (BTC)", 0.01, 10.0, 0.1)
        order_type = st.selectbox("Type", ["Market", "Limit"])
        price = st.number_input("Price", disabled=(order_type == "Market")) if order_type == "Limit" else None
        stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.01)
        take_profit = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.01)
        
        if st.form_submit_button("🚀 Execute Order"):
            try:
                order_params = {
                    "symbol": symbol,
                    "side": "Buy" if side == "Buy" else "Sell",
                    "orderQty": quantity,
                    "ordType": order_type,
                    "price": price
                }
                if TESTNET_MODE:
                    result = virtual_phemex_exchange.simulate_testnet_order(symbol, order_params)
                    st.success(f"Order ID: {result['data']['orderID']}, New Balance: {result['data']['mUSDBalance']:.2f} mUSD")
                else:
                    result = asyncio.run(client.place_order(order_params))
                    st.success(f"Order ID: {result['orderID']}")
            except PhemexAPIError as e:
                st.error(f"Order failed: {str(e)}")
            except ValueError as e:
                st.error(f"Simulation error: {str(e)}")

# Risk Management Controls
st.sidebar.header("🛡️ Risk Management")
with st.sidebar.expander("Position Limits"):
    st.slider("Max BTC Exposure", 0.1, 5.0, 1.0)
    st.selectbox("Leverage Mode", ["1x", "3x", "5x", "10x"])

with st.sidebar.expander("Circuit Breakers"):
    st.number_input("Daily Loss Limit (%)", 1, 100, 5)
    st.number_input("Max Orders/Min", 1, 100, 15)

# Performance Metrics
st.header("Performance Metrics")
st.write("Metrics will be displayed here.")

# Monitoring and Alerts
st.header("Monitoring & Alerts")
st.write("Alerts will be displayed here.")

# Footer
st.write("Developed by [Your Name]")
