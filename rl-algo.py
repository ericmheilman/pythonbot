import random
import backtrader as bt
import pandas as pd
import logging
import requests
from pycoingecko import CoinGeckoAPI
import numpy as np
from stable_baselines3 import DQN  # DQN RL algorithm from Stable Baselines3
from gym import Env, spaces

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# File handler for verbose logging
file_handler = logging.FileHandler('backtest.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# API Key for CoinGecko (replace with your own)
API_KEY = 'CG-SwSx6aTNiJDmG1TiXVMGHPbg'

# Placeholder to store profitable strategies
profitable_strategies_db = []

# Strategy categories
strategy_categories = {
    "Scalping": {"timeframes": ['1m', '5m'], "history_limit": 500},
    "Swing Trading": {"timeframes": ['1h', '4h'], "history_limit": 500},
    "Position Trading": {"timeframes": ['1d', '1w'], "history_limit": 500},
    "Day Trading": {"timeframes": ['15m', '1h'], "history_limit": 500},
}

# Function to pull data for multiple timeframes
def fetch_multi_timeframe_data(symbol, vs_currency='usd', timeframes=['1', '7', '30']):
    cg = CoinGeckoAPI()
    
    # Convert symbol to lowercase to match CoinGecko format
    symbol = symbol.lower()

    dfs = []
    for days in timeframes:
        logger.debug(f"Fetching {days} days of data for {symbol}")
        
        # Fetch historical market data from CoinGecko
        historical_data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency=vs_currency, days=days)
        
        # Create a DataFrame from the price data
        df = pd.DataFrame(historical_data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Add open, high, low, close, and volume columns to match Backtrader's expected format
        df['open'] = df['price']
        df['high'] = df['price']
        df['low'] = df['price']
        df['close'] = df['price']
        df['volume'] = 1  # Placeholder

        dfs.append(df)

    # Combine the dataframes from multiple timeframes into one DataFrame
    combined_df = pd.concat(dfs, axis=0).sort_index()
    return combined_df

# Define RL environment for trading
class TradingEnv(Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.starting_cash = 10000
        self.cash = self.starting_cash
        self.positions = 0

        # Define the action space (Buy, Sell, Hold)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space is a vector of prices + indicators (normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.starting_cash
        self.positions = 0
        return self._get_observation()

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        if action == 1:  # Buy
            if self.cash >= current_price:
                self.positions += 1
                self.cash -= current_price
                logger.debug(f"Buy at {current_price}")
        
        elif action == 2:  # Sell
            if self.positions > 0:
                self.positions -= 1
                self.cash += current_price
                logger.debug(f"Sell at {current_price}")
                reward = (self.cash + (self.positions * current_price)) - self.starting_cash

        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        # Calculate net worth
        net_worth = self.cash + self.positions * current_price
        reward += (net_worth - self.starting_cash) / self.starting_cash  # Reward as a return on investment

        return obs, reward, done, {}

    def _get_observation(self):
        obs = [
            self.df.iloc[self.current_step]['open'],
            self.df.iloc[self.current_step]['high'],
            self.df.iloc[self.current_step]['low'],
            self.df.iloc[self.current_step]['close'],
            self.positions,
            self.cash / self.starting_cash  # Normalize cash
        ]
        return np.array(obs, dtype=np.float32)

# Backtest strategy using RL
def backtest_rl_strategy(df, symbol):
    env = TradingEnv(df)
    model = DQN('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=50000)

    # Run the model to test performance
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

    net_worth = env.cash + env.positions * df.iloc[env.current_step]['close']
    profit = net_worth - env.starting_cash
    logger.info(f"Profit for {symbol}: {profit}")
    return profit

# Main function to run the backtesting with RL
def main():
    logger.info("Starting Reinforcement Learning strategy backtesting...")
    symbol = 'solana'  # Replace with desired asset

    # Fetch data for the symbol
    df = fetch_multi_timeframe_data(symbol=symbol, timeframes=['1', '7', '30'])

    # Backtest the strategy using RL
    profit = backtest_rl_strategy(df, symbol)
    logger.info(f"Final profit from RL strategy: {profit}")

if __name__ == "__main__":
    main()
