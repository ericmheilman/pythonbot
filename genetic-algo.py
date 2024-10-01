import random
import backtrader as bt
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Example trading categories
strategy_categories = {
    "Scalping": {"timeframe": "1m", "history_limit": 5000},
    "Swing Trading": {"timeframe": "1h", "history_limit": 1000},
    "Position Trading": {"timeframe": "1d", "history_limit": 365},
    "Day Trading": {"timeframe": "15m", "history_limit": 2000},
}

# Predefined strategies
common_strategies = [
    {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_overbought': 70, 'rsi_oversold': 30, 'adx_threshold': 25, 'target_profit': 1.5, 'atr_multiplier': 2.0},
    {'macd_fast': 8, 'macd_slow': 26, 'macd_signal': 9, 'rsi_overbought': 65, 'rsi_oversold': 35, 'adx_threshold': 25, 'target_profit': 2, 'atr_multiplier': 2.5},
]

# Fetch price data
def fetch_data(symbol, timeframe, limit):
    # For demonstration, replace this with real price data fetching logic
    dates = pd.date_range(start="2023-01-01", periods=limit, freq='T')
    prices = np.random.normal(loc=20, scale=5, size=limit)  # Fake price data
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": prices, "low": prices, "close": prices})
    df.set_index("timestamp", inplace=True)
    return df

# Evaluate strategy using backtest results
def evaluate_strategy(individual):
    # Ensure periods are valid
    macd_fast = max(1, individual[0])
    macd_slow = max(macd_fast + 1, individual[1])
    macd_signal = max(1, individual[2])
    rsi_overbought = individual[3]
    rsi_oversold = individual[4]
    adx_threshold = individual[5]
    target_profit = individual[6]
    atr_multiplier = individual[7]
    
    strategy = {
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal,
        'rsi_overbought': rsi_overbought,
        'rsi_oversold': rsi_oversold,
        'adx_threshold': adx_threshold,
        'target_profit': target_profit,
        'atr_multiplier': atr_multiplier,
    }
    
    df = fetch_data(symbol='SOL/USDT', timeframe='1m', limit=5000)
    profit = backtest_strategy(strategy, df)
    return profit,

# Backtest function for strategy profitability
def backtest_strategy(strategy, df):
    cerebro = bt.Cerebro()

    class TestStrategy(bt.Strategy):
        def __init__(self):
            self.macd = bt.indicators.MACD(period_me1=strategy['macd_fast'], period_me2=strategy['macd_slow'], period_signal=strategy['macd_signal'])
            self.rsi = bt.indicators.RSI_Safe()
            self.adx = bt.indicators.ADX()
            self.atr = bt.indicators.ATR()

        def next(self):
            if self.rsi[0] < strategy['rsi_oversold'] and self.adx[0] > strategy['adx_threshold']:
                self.buy()
            elif self.rsi[0] > strategy['rsi_overbought'] and self.adx[0] > strategy['adx_threshold']:
                self.sell()

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()

    final_value = cerebro.broker.getvalue()
    return final_value - 10000  # Return profit

# Genetic Algorithm setup
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
    toolbox.register("attr_target_profit", random.uniform, 1.5, 3.0)
    toolbox.register("attr_atr_multiplier", random.uniform, 1.5, 3.0)

    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_macd_fast, toolbox.attr_macd_slow, toolbox.attr_macd_signal, 
                      toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold, toolbox.attr_adx_threshold, 
                      toolbox.attr_target_profit, toolbox.attr_atr_multiplier), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

# Main function to run GA
def main():
    toolbox = setup_ga()
    population = toolbox.population(n=100)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=True)

    # Rank strategies by profitability
    ranked_strategies = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)

    # Print top strategies
    print("\nTop 10 Strategies:")
    for i, strategy in enumerate(ranked_strategies[:10], 1):
        print(f"Rank {i}: Strategy {strategy}, Profit: {strategy.fitness.values[0]}")

if __name__ == "__main__":
    main()
