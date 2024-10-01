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

# File handler for logging to a file
file_handler = logging.FileHandler('backtest.log')
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

# Function to fetch data (this uses dummy data for backtesting purposes)
def fetch_data(symbol, timeframe, limit):
    freq_map = {'1m': 'T', '15m': '15T', '1h': 'H', '1d': 'D'}
    dates = pd.date_range(start="2023-01-01", periods=limit, freq=freq_map[timeframe])
    prices = np.random.normal(loc=100, scale=10, size=limit)
    
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": prices * 1.02, "low": prices * 0.98, "close": prices})
    df.set_index("timestamp", inplace=True)
    
    return df

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
            ('stoch_k', 14),
            ('stoch_d', 3),
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
            self.stochastic = bt.indicators.Stochastic(
                self.data, 
                period=self.params.stoch_k, 
                period_dfast=self.params.stoch_d
            )

        def next(self):
            if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
                self.buy()
            elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
                self.sell()

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

# Define the GA for optimizing the strategy
def setup_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_macd_fast", random.randint, 6, 15)
    toolbox.register("attr_macd_slow", random.randint, 20, 30)
    toolbox.register("attr_macd_signal", random.randint, 5, 12)
    toolbox.register("attr_rsi_overbought", random.randint, 60, 80)
    toolbox.register("attr_rsi_oversold", random.randint, 20, 40)
    toolbox.register("attr_adx_threshold", random.randint, 20, 35)
    toolbox.register("attr_ema_period", random.randint, 5, 50)
    toolbox.register("attr_bb_period", random.randint, 5, 50)
    toolbox.register("attr_stoch_k", random.randint, 5, 20)
    toolbox.register("attr_stoch_d", random.randint, 3, 10)

    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_macd_fast, toolbox.attr_macd_slow, toolbox.attr_macd_signal, 
                      toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold, toolbox.attr_adx_threshold, 
                      toolbox.attr_ema_period, toolbox.attr_bb_period,
                      toolbox.attr_stoch_k, toolbox.attr_stoch_d), n=1)

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
        'ema_period': individual[6],
        'bb_period': individual[7],
        'stoch_k': individual[8],
        'stoch_d': individual[9],
    }
    
    df = fetch_data(symbol='SOL/USDT', timeframe='1m', limit=5000)
    profit = backtest_strategy(strategy, df)
    
    return profit,

# Run GA and backtest for each trading category
def run_category_backtests():
    all_results = []

    for category, params in strategy_categories.items():
        print(f"\nRunning backtests for {category}...")

        df = fetch_data(symbol='SOL/USDT', timeframe=params['timeframe'], limit=params['history_limit'])

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
                'ema_period': ind[6],
                'bb_period': ind[7],
                'stoch_k': ind[8],
                'stoch_d': ind[9],
            }
            profit = backtest_strategy(strategy, df)
            all_results.append((strategy, profit))

    return all_results

# Main function to run everything
def main():
    all_results = run_category_backtests()

    # Prepare data for AI model training
    df = pd.DataFrame([r[0] for r in all_results])
    df['profit'] = [r[1] for r in all_results]

    X = df.drop(columns=['profit'])
    y = df['profit']

    ai_results = train_ai_models(X, y)

    # Print AI model results
    print("\nAI Model Performance:")
    for model_name, score in ai_results.items():
        print(f"{model_name}: {score}")

if __name__ == "__main__":
    main()
