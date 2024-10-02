import random
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from pycoingecko import CoinGeckoAPI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# File handler for verbose logging
file_handler = logging.FileHandler('NSR-backtest.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Example trading categories with timeframes
strategy_categories = {
    "Scalping": {"timeframe": "1m", "history_limit": 5000},
    "Swing Trading": {"timeframe": "1h", "history_limit": 1000},
    "Position Trading": {"timeframe": "1d", "history_limit": 365},
    "Day Trading": {"timeframe": "15m", "history_limit": 2000},
}

# Function to pull data for a specific timeframe
def fetch_data(symbol, timeframe='1m', limit=5000):
    cg = CoinGeckoAPI()
    symbol = symbol.lower()
    
    logger.debug(f"Fetching {limit} data points for {symbol} over {timeframe}")
    
    try:
        # Fetch historical market data from CoinGecko
        historical_data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=limit // 1440)  # Roughly convert limit to days

        # Create a DataFrame from the price data
        df = pd.DataFrame(historical_data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Add open, high, low, close, and volume columns to match Backtrader's expected format
        df['open'] = df['price']
        df['high'] = df['price'] + np.random.uniform(0.1, 1, len(df))  # Simulate high values
        df['low'] = df['price'] - np.random.uniform(0.1, 1, len(df))  # Simulate low values
        df['close'] = df['price']
        df['volume'] = 1  # Placeholder for volume

        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Backtest a strategy using Backtrader
def backtest_strategy(strategy, df):
    cerebro = bt.Cerebro()

    class TestStrategy(bt.Strategy):
        params = (
            ('macd_fast', 12),
            ('macd_slow', 26),
            ('macd_signal', 9),
            ('rsi_overbought', 70),
            ('rsi_oversold', 30),
            ('adx_threshold', 25),
            ('ema_period', 20),
            ('bb_period', 20),
        )

        def __init__(self):
            self.macd = bt.indicators.MACD(
                period_me1=self.params.macd_fast,
                period_me2=self.params.macd_slow,
                period_signal=self.params.macd_signal
            )
            self.rsi = bt.indicators.RSI_Safe()
            self.bb = bt.indicators.BollingerBands(period=self.params.bb_period)
            self.adx = bt.indicators.ADX()
            self.ema = bt.indicators.EMA(period=self.params.ema_period)

        def next(self):
            if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
                self.buy()
            elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
                self.sell()

            # Mean reversion using Bollinger Bands
            if self.data.close[0] > self.bb.lines.top[0]:
                self.sell()
            elif self.data.close[0] < self.bb.lines.bot[0]:
                self.buy()

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()
    
    return cerebro.broker.getvalue() - 10000  # Return profit

# Train and test AI models on the strategy results
def train_ai_models(X, y):
    models = {
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500),
        "SVM": SVR(),
        "Random Forest": RandomForestRegressor(),
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results[model_name] = score
    
    return results

# Define an evolutionary algorithm (GA) to tweak strategies
def setup_ga():
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_macd_fast", random.randint, 6, 15)
    toolbox.register("attr_macd_slow", random.randint, 20, 30)
    toolbox.register("attr_macd_signal", random.randint, 5, 12)
    toolbox.register("attr_rsi_overbought", random.randint, 60, 80)
    toolbox.register("attr_rsi_oversold", random.randint, 20, 40)
    toolbox.register("attr_adx_threshold", random.randint, 20, 35)
    toolbox.register("attr_target_profit", random.uniform, 1.5, 3.0)
    toolbox.register("attr_atr_multiplier", random.uniform, 1.5, 3.0)
    toolbox.register("attr_ema_period", random.randint, 5, 50)
    toolbox.register("attr_bb_period", random.randint, 5, 50)

    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_macd_fast, toolbox.attr_macd_slow, toolbox.attr_macd_signal, 
                      toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold, toolbox.attr_adx_threshold, 
                      toolbox.attr_target_profit, toolbox.attr_atr_multiplier,
                      toolbox.attr_ema_period, toolbox.attr_bb_period), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_ga_strategy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

# Evaluate strategy generated by GA
def evaluate_ga_strategy(individual):
    strategy = {
        'macd_fast': individual[0],
        'macd_slow': individual[1],
        'macd_signal': individual[2],
        'rsi_overbought': individual[3],
        'rsi_oversold': individual[4],
        'adx_threshold': individual[5],
        'target_profit': individual[6],
        'atr_multiplier': individual[7],
        'ema_period': individual[8],
        'bb_period': individual[9],
    }
    
    # Fetch and backtest strategy
    df = fetch_data(symbol='solana', limit=5000)
    profit = backtest_strategy(strategy, df)
    
    return profit,

# Run AI algorithms for each trading category
def run_category_backtests():
    all_results = []

    for category, params in strategy_categories.items():
        print(f"\nRunning backtests for {category}...")

        # Fetch price data for the category
        df = fetch_data(symbol='solana', timeframe=params['timeframe'], limit=params['history_limit'])

        # Create and test strategies using GA
        toolbox = setup_ga()
        population = toolbox.population(n=100)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

        for ind in population:
            strategy = {
                'macd_fast': ind[0],
                'macd_slow': ind[1],
                'macd_signal': ind[2],
                'rsi_overbought': ind[3],
                'rsi_oversold': ind[4],
                'adx_threshold': ind[5],
                'target_profit': ind[6],
                'atr_multiplier': ind[7],
                'ema_period': ind[8],
                'bb_period': ind[9],
            }
            profit = backtest_strategy(strategy, df)
            all_results.append((strategy, profit))

    return all_results

# Main function
def main():
    all_results = run_category_backtests()

    # Prepare data for AI model training
    df = pd.DataFrame([r[0] for r in all_results])
    df['profit'] = [r[1] for r in all_results]

    X = df.drop(columns=['profit'])
    y = df['profit']

    # Train AI models
    ai_results = train_ai_models(X, y)

    # Print AI model results
    print("\nAI Model Performance:")
    for model_name, score in ai_results.items():
        print(f"{model_name}: {score}")

if __name__ == "__main__":
    main()
