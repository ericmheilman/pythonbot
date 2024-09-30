import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import random
from sklearn.model_selection import ParameterGrid

# Define the AI models to iterate through
AI_MODELS = ["Genetic Algorithm", "Reinforcement Learning", "Neural Network", "Random Forest", "Decision Tree"]

# Define categories and respective backtesting timeframes
TRADING_CATEGORIES = {
    "scalping": "5m",  # Short-term trading
    "swing": "1h",     # Medium-term trading
    "position": "1d",  # Long-term trading
    "day_trading": "15m",  # Example of intraday trading
    "momentum_trading": "30m",  # Example of momentum trading
    "mean_reversion": "1h",     # Example of mean reversion trading
    "trend_following": "4h",    # Trend-following strategy, more long-term
    "breakout": "1h"            # Breakout trading strategy
}

# Strategies database for top 100 strategies
strategies_database = []

# Step 1: Fetch real-time data from Binance (or other exchange) for the respective timeframe
exchange = ccxt.binance()

def fetch_data(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Step 2: Calculate technical indicators
def calculate_indicators(df):
    bb = ta.bbands(df['close'], length=20, std=2)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    
    df = pd.concat([df, macd], axis=1)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df

# Step 3: Trading strategy logic based on params
class DynamicStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('adx_threshold', 25),
        ('target_profit', 2.0),
        ('atr_multiplier', 1.5)
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            period_me1=self.params.macd_fast, 
            period_me2=self.params.macd_slow, 
            period_signal=self.params.macd_signal
        )
        self.bbands = bt.indicators.BollingerBands(period=20, devfactor=2)
        self.rsi = bt.indicators.RSI_Safe()
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()

    def next(self):
        if self.data.close[0] > self.bbands.lines.top[0] and self.rsi[0] < self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.buy()

        if self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] > self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
            self.sell()

# Step 4: Backtest strategy
def backtest_strategy(params, df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    cerebro.addstrategy(DynamicStrategy, 
                        macd_fast=int(params['macd_fast']), 
                        macd_slow=int(params['macd_slow']), 
                        macd_signal=int(params['macd_signal']), 
                        rsi_overbought=int(params['rsi_overbought']), 
                        rsi_oversold=int(params['rsi_oversold']), 
                        adx_threshold=int(params['adx_threshold']), 
                        target_profit=params['target_profit'], 
                        atr_multiplier=params['atr_multiplier'])

    cerebro.broker.set_cash(10000)
    cerebro.run()

    final_value = cerebro.broker.getvalue()
    return final_value

# Step 5: Generate new strategies
def generate_new_strategy(category):
    param_grid = {
        'macd_fast': [8, 12, 16],
        'macd_slow': [26, 30, 35],
        'macd_signal': [9],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
        'adx_threshold': [20, 25, 30],
        'target_profit': [1.5, 2, 2.5],
        'atr_multiplier': [1.5, 2.0, 2.5]
    }

    param_combinations = list(ParameterGrid(param_grid))
    strategy = random.choice(param_combinations)
    return strategy

# Step 6: AI Model Switching Logic
def ai_model_switch(model_index):
    print(f"Using AI Model: {AI_MODELS[model_index]}")

# Step 7: Main process to run for each category, AI model, and strategy
def main():
    global strategies_database

    for category, timeframe in TRADING_CATEGORIES.items():
        print(f"Processing category: {category}, with timeframe: {timeframe}")
        
        df = fetch_data('SOL/USDT', timeframe)
        df = calculate_indicators(df)

        # Iterate through AI models
        for model_index, ai_model in enumerate(AI_MODELS):
            ai_model_switch(model_index)
            
            # Generate 10 strategies for this category
            for _ in range(10):
                new_strategy = generate_new_strategy(category)
                result = backtest_strategy(new_strategy, df)
                
                strategies_database.append({'category': category, 'strategy': new_strategy, 'performance': result})

                # Sort strategies based on profitability and maintain top 100
                strategies_database = sorted(strategies_database, key=lambda x: x['performance'], reverse=True)[:100]

    # Output top 20 strategies
    for i, strategy_info in enumerate(strategies_database[:20]):
        print(f"Rank {i+1}: {strategy_info}")

if __name__ == "__main__":
    main()
