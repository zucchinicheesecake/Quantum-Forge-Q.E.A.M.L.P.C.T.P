# Import all necessary libraries
import time 
import config
import ccxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from textblob import TextBlob  # For sentiment analysis

# Bot connection
phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': config.API_KEY,
    'secret': config.API_SECRET,
})

# Reinforcement Learning Agent Class
class ReinforcementLearningAgent:
    def __init__(self, state_size=8, action_size=4):
        self.model = self._build_dqn_model(state_size, action_size)

    def _build_dqn_model(self, state_size, action_size):
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def train(self, states, actions, rewards, next_states, done):
        target = rewards + 0.95 * np.amax(self.model.predict(next_states), axis=1) * (1 - done)
        target_f = self.model.predict(states)
        target_f[0][actions] = target
        self.model.fit(states, target_f, epochs=1, verbose=0)

# Opportunity Detection Functions
def detect_micro_arbitrage(pair1: str, pair2: str) -> float:
    """Find 0.1-0.8% spreads between correlated pairs"""
    price1 = phemex.fetch_ticker(pair1)['last']
    price2 = phemex.fetch_ticker(pair2)['last']
    return abs((price1 - price2) / price1) * 100  # Spread percentage

def analyze_sentiment(symbol: str) -> float:
    """Real-time Twitter sentiment scoring"""
    tweets = self.twitter_api.search(q=f"${symbol.split('/')[0]}", count=100)
    sentiment_scores = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets]
    return np.mean(sentiment_scores)  # -1 (bearish) to +1 (bullish)
