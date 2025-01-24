import streamlit as st
from phemex_api import PhemexTradingCore, PhemexAPIError
import asyncio

# Add to sidebar controls
with st.sidebar:
    st.subheader("Phemex Connection")
    if st.button("🔌 Test API Connection"):
        try:
            client = PhemexTradingCore()
            orderbook = asyncio.run(client.get_orderbook())
            st.success(f"✅ Connected - BTC/USD Bid: {orderbook['book']['bids'][0][0]}")
        except PhemexAPIError as e:
            st.error(f"Connection failed: {str(e)}")

# Add to deployment_dashboard()
def deployment_dashboard():
    client = PhemexTradingCore()
    
    # Real-time data column
    col1, col2, col3 = st.columns(3)
    with col1:
        orderbook = asyncio.run(client.get_orderbook())
        st.metric("BTC/USD Bid-Ask", 
                f"{orderbook['book']['bids'][0][0]} | {orderbook['book']['asks'][0][0]}")

    # Order entry form
    with st.expander("📈 Place Manual Order"):
        with st.form("order_form"):
            symbol = st.selectbox("Pair", ["BTCUSD", "ETHUSD"])
            side = st.radio("Side", ["Buy", "Sell"])
            quantity = st.number_input("Quantity (BTC)", 0.01, 10.0, 0.1)
            order_type = st.selectbox("Type", ["Market", "Limit"])
            price = st.number_input("Price", disabled=(order_type == "Market")) if order_type == "Limit" else None
            
            if st.form_submit_button("🚀 Execute Order"):
                try:
                    order_params = {
                        "symbol": symbol,
                        "side": "Buy" if side == "Buy" else "Sell",
                        "orderQty": quantity,
                        "ordType": order_type,
                        "price": price
                    }
                    result = asyncio.run(client.place_order(order_params))
                    st.success(f"Order ID: {result['orderID']}")
                except PhemexAPIError as e:
                    st.error(f"Order failed: {str(e)}")
