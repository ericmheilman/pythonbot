import random
import backtrader as bt
import pandas as pd
import logging
from pycoingecko import CoinGeckoAPI
import numpy as np
from stable_baselines3 import PPO
from gym import Env, spaces
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# File handler for verbose logging
file_handler = logging.FileHandler('RL-backtest.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# CoinGecko API configuration
API_KEY = 'CG-SwSx6aTNiJDmG1TiXVMGHPbg'

# Strategy categories (multi-timeframe)
strategy_categories = {
    "Scalping": {"timeframes": ['1m', '5m'], "history_limit": 500},
    "Swing Trading": {"timeframes": ['1h', '4h'], "history_limit": 500},
    "Position Trading": {"timeframes": ['1d', '1w'], "history_limit": 500},
    "Day Trading": {"timeframes": ['15m', '1h'], "history_limit": 500},
}
def fetch_multi_timeframe_data(symbol, vs_currency='usd', timeframes=['1', '7', '30']):
    cg = CoinGeckoAPI()
    
    symbol = symbol.lower()
    dfs = []
    
    for days in timeframes:
        logger.debug(f"Fetching {days} days of data for {symbol}")
        historical_data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency=vs_currency, days=days)
        df = pd.DataFrame(historical_data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['open'] = df['price']
        df['high'] = df['price'] * 1.02
        df['low'] = df['price'] * 0.98
        df['close'] = df['price']
        df['volume'] = random.uniform(0.1, 10)  # Placeholder for volume
        
        # Adding indicators and filling missing values
        df['sma_slow'] = SMAIndicator(close=df['close'], window=50).sma_indicator().fillna(0)
        df['sma_fast'] = SMAIndicator(close=df['close'], window=20).sma_indicator().fillna(0)
        df['macd'] = MACD(close=df['close']).macd_diff().fillna(0)
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx().fillna(0)
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi().fillna(0)
        df['stoch'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14).stoch().fillna(0)
        df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r().fillna(0)
        df['bollinger_high'] = BollingerBands(close=df['close'], window=20, window_dev=2).bollinger_hband().fillna(0)
        df['bollinger_low'] = BollingerBands(close=df['close'], window=20, window_dev=2).bollinger_lband().fillna(0)
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range().fillna(0)
        df['donchian_high'] = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20).donchian_channel_hband().fillna(0)
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume().fillna(0)
        df['mfi'] = calculate_mfi(df).fillna(0)
        df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).volume_weighted_average_price().fillna(0)

        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0).sort_index()
    return combined_df


# Manual Money Flow Index (MFI) calculation
def calculate_mfi(df, window=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # Positive and Negative Money Flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    # Money Flow Ratio
    money_flow_ratio = positive_flow.rolling(window=window).sum() / negative_flow.rolling(window=window).sum()

    # Money Flow Index
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

# Define RL environment for trading with indicators
class TradingEnv(Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.starting_cash = 10000
        self.cash = self.starting_cash
        self.positions = 0

        # Action space: 0: hold, 1: buy, 2: sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: prices and technical indicators
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32  # Added 15 features including technical indicators
        )

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
        reward += (net_worth - self.starting_cash) / self.starting_cash

        # Add reward based on drawdown to penalize high-risk behavior
        if reward < 0:
            reward -= 0.01 * abs(reward)

        return obs, reward, done, {}

    def _get_observation(self):
        obs = [
            self.df.iloc[self.current_step]['open'],
            self.df.iloc[self.current_step]['high'],
            self.df.iloc[self.current_step]['low'],
            self.df.iloc[self.current_step]['close'],
            self.df.iloc[self.current_step]['volume'],
            self.df.iloc[self.current_step]['sma_slow'],    # Slow moving average
            self.df.iloc[self.current_step]['rsi'],          # RSI
            self.df.iloc[self.current_step]['bollinger_high'], # Bollinger High Band
            self.df.iloc[self.current_step]['atr'],           # ATR
            self.df.iloc[self.current_step]['obv'],           # On Balance Volume
            self.df.iloc[self.current_step]['macd'],          # MACD
            self.df.iloc[self.current_step]['stoch'],         # Stochastic Oscillator
            self.df.iloc[self.current_step]['vwap'],          # VWAP
            self.df.iloc[self.current_step]['adx'],           # ADX
            self.df.iloc[self.current_step]['mfi'],           # Money Flow Index
        ]
        return np.array(obs, dtype=np.float32)

# Backtest strategy using RL
def backtest_rl_strategy(df, symbol):
    env = TradingEnv(df)
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model with extended steps
    model.learn(total_timesteps=200000)

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
