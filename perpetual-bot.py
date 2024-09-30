import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt
from sklearn.model_selection import ParameterGrid
import random
import json
import os

# --- Step 1: Fetch data ---
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# --- Step 2: Calculate Indicators ---
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

# --- Step 3: Trading Strategy Class ---
class DynamicStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('adx_threshold', 25),
        ('target_profit', 1.5),
        ('atr_multiplier', 2.5)
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

# --- Step 4: Backtesting Function ---
def backtest_strategy(params, df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Ensure that period parameters are integers, such as macd_fast, macd_slow, etc.
    cerebro.addstrategy(DynamicStrategy, 
                        macd_fast=int(params['macd_fast']),  # Cast to int
                        macd_slow=int(params['macd_slow']),  # Cast to int
                        macd_signal=int(params['macd_signal']),  # Cast to int
                        rsi_overbought=int(params['rsi_overbought']),  # Cast to int
                        rsi_oversold=int(params['rsi_oversold']),  # Cast to int
                        adx_threshold=int(params['adx_threshold']),  # Cast to int
                        target_profit=params['target_profit'],  # This can remain a float
                        atr_multiplier=params['atr_multiplier']  # This can remain a float
                       )
    
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.001)

    # Run the backtest
    cerebro.run()
    
    # Get the final portfolio value as the result of the strategy
    final_value = cerebro.broker.getvalue()

    return final_value


# --- Step 5: AI Model Rotation & Strategy Creation ---
# Define AI Models to be used
ai_models = ['Genetic Algorithm', 'Reinforcement Learning', 'Neural Network', 'GPT-Style', 'Random Forest', 'Gradient Boosting', 'AutoML']

def rotate_ai_model(ai_models, strategy_count):
    return ai_models[strategy_count % len(ai_models)]

# --- AI Model-specific Strategy Generation Logic ---
def create_ai_strategies(model_name):
    strategies = []
    if model_name == 'Genetic Algorithm':
        # Simulate crossover and mutation of existing strategies
        for _ in range(10):
            params = random.choice(common_strategies)
            mutated_params = {key: val * (1 + random.uniform(-0.1, 0.1)) for key, val in params.items()}
            strategies.append(mutated_params)

    elif model_name == 'Reinforcement Learning':
        # Use state-action-reward optimization for strategy creation (simplified here)
        for _ in range(10):
            params = random.choice(common_strategies)
            params['target_profit'] += random.uniform(0, 0.5)  # Adjust target profit
            params['rsi_overbought'] += random.randint(-5, 5)
            strategies.append(params)

    elif model_name == 'Neural Network':
        # Neural network models try different combinations based on patterns
        for _ in range(10):
            param_grid = {
                'macd_fast': [8, 12, 14],
                'macd_slow': [26, 30],
                'macd_signal': [9, 12],
                'rsi_overbought': [65, 70, 75],
                'rsi_oversold': [25, 30, 35],
                'adx_threshold': [20, 25, 30],
                'target_profit': [1.5, 2.0, 2.5],
                'atr_multiplier': [2.0, 2.5]
            }
            for params in ParameterGrid(param_grid):
                strategies.append(params)

    elif model_name == 'GPT-Style':
        # GPT model tweaks parameters creatively
        for _ in range(10):
            params = random.choice(common_strategies)
            params['rsi_overbought'] = random.choice([60, 65, 70, 75, 80])
            params['target_profit'] = random.uniform(1.0, 2.5)
            strategies.append(params)

    elif model_name == 'Random Forest':
        # Decision tree models split and refine the params
        for _ in range(10):
            params = random.choice(common_strategies)
            params['adx_threshold'] = random.choice([20, 25, 30])
            params['target_profit'] += random.uniform(-0.5, 0.5)
            strategies.append(params)

    elif model_name == 'Gradient Boosting':
        # Boosting model tweaks based on patterns
        for _ in range(10):
            params = random.choice(common_strategies)
            params['atr_multiplier'] += random.uniform(-0.5, 0.5)
            strategies.append(params)

    elif model_name == 'AutoML':
        # AutoML-based strategy creation
        param_grid = {
            'macd_fast': [8, 12, 14],
            'macd_slow': [26, 30],
            'macd_signal': [9, 12],
            'rsi_overbought': [65, 70, 75],
            'rsi_oversold': [25, 30, 35],
            'adx_threshold': [20, 25, 30],
            'target_profit': [1.5, 2.0, 2.5],
            'atr_multiplier': [2.0, 2.5]
        }
        strategies = list(ParameterGrid(param_grid))

    return strategies

# --- Step 6: Main AI-driven Strategy Generation and Testing ---
def main():
    df = fetch_data()
    df = calculate_indicators(df)

    strategy_db = []

    # Add traditional strategies first
    for strategy in common_strategies:
        result = backtest_strategy(strategy, df)
        strategy['result'] = result
        strategy_db.append(strategy)

    strategy_db = sorted(strategy_db, key=lambda x: x['result'], reverse=True)

    strategy_count = 0
    while True:
        ai_model = rotate_ai_model(ai_models, strategy_count)
        print(f"Using AI Model: {ai_model}")

        new_strategies = create_ai_strategies(ai_model)

        for strategy in new_strategies:
            result = backtest_strategy(strategy, df)
            strategy['result'] = result
            strategy_db.append(strategy)

            # Maintain the top 100 strategies
            strategy_db = sorted(strategy_db, key=lambda x: x['result'], reverse=True)[:100]

            save_strategies_to_file(strategy_db)

        strategy_count += 1

def save_strategies_to_file(strategy_db):
    with open('strategy_db.json', 'w') as f:
        json.dump(strategy_db, f)

if __name__ == "__main__":
    main()
