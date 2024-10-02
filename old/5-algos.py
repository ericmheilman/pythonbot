import random
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor

# Example trading categories with reduced data limits
strategy_categories = {
    "Scalping": {"timeframe": "1m", "history_limit": 1000},
    "Swing Trading": {"timeframe": "1h", "history_limit": 500},
    "Position Trading": {"timeframe": "1d", "history_limit": 100},
    "Day Trading": {"timeframe": "15m", "history_limit": 500},
}

# Placeholder to store profitable strategies
profitable_strategies_db = []

# Algorithms to iterate over
algorithms_list = ['Genetic Algorithm', 'Random Forest', 'Neural Network', 'SVM', 'Decision Tree',
                   'Gradient Boosting', 'AdaBoost', 'KNN', 'XGBoost', 'Logistic Regression']

def fetch_real_data(symbol, timeframe, limit):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Replace your fetch_data function with this:
df = fetch_real_data(symbol='SOL/USDT', timeframe='1m', limit=5000)
profit = backtest_strategy(strategy, df)
print(f"Profit: {profit}")


# Pull historical data for ETH, BTC, and SOL
def pull_historical_data():
    sol_data = fetch_data('SOL/USDT', '1m', 1000)
    eth_data = fetch_data('ETH/USDT', '1m', 1000)
    btc_data = fetch_data('BTC/USDT', '1m', 1000)
    return {'SOL': sol_data, 'ETH': eth_data, 'BTC': btc_data}

# Backtest a strategy using Backtrader
def backtest_strategy(strategy, df):
    cerebro = bt.Cerebro()
    class TestStrategy(bt.Strategy):
        params = (
            ('macd_fast', 12),   # Fast MACD period
            ('macd_slow', 26),   # Slow MACD period
            ('macd_signal', 9),  # Signal line period
            ('rsi_overbought', 70),
            ('rsi_oversold', 30),
            ('adx_threshold', 25)
        )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            period_me1=self.params.macd_fast, 
            period_me2=self.params.macd_slow, 
            period_signal=self.params.macd_signal
        )
        self.rsi = bt.indicators.RSI_Safe()
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()

    def next(self):
        if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
            self.buy()
            print(f"Buy at {self.data.close[0]} on {self.data.datetime.datetime(0)}")
        elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.sell()
            print(f"Sell at {self.data.close[0]} on {self.data.datetime.datetime(0)}")

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()

    return cerebro.broker.getvalue() - 10000  # Return profit

# Train and test AI models on the strategy results
def train_ai_models(X, y):
    models = {
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200),
        "SVM": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(),
        "Logistic Regression": LogisticRegression(),  # Though this is a classification model, it's included for flexibility
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
    }
    
    df = fetch_data(symbol='SOL/USDT', timeframe='1m', limit=1000)
    profit = backtest_strategy(strategy, df)
    
    return profit,

# Run AI algorithms for each trading category and each algorithm
def run_algorithm_backtests(symbol, category, params):
    results = []
    df = fetch_data(symbol=symbol, timeframe=params['timeframe'], limit=params['history_limit'])

    for algo in algorithms_list:
        for model_num in range(3):  # Test 3 models per algorithm
            if algo == 'Genetic Algorithm':
                # Run Genetic Algorithm
                toolbox = setup_ga()
                population = toolbox.population(n=10)
                algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=3, verbose=False)

                # Rank strategies by profitability
                ranked_strategies = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
                for strategy in ranked_strategies[:1]:  # Top strategy
                    profitable_strategies_db.append(('Genetic Algorithm', strategy, strategy.fitness.values[0]))
            else:
                # Random strategy for ML models
                strategy = {
                    'macd_fast': random.randint(6, 15),
                    'macd_slow': random.randint(20, 30),
                    'macd_signal': random.randint(5, 12),
                    'rsi_overbought': random.randint(60, 80),
                    'rsi_oversold': random.randint(20, 40),
                    'adx_threshold': random.randint(20, 35),
                    'target_profit': random.uniform(1.5, 3.0),
                    'atr_multiplier': random.uniform(1.5, 3.0),
                }

                # Backtest and store results
                profit = backtest_strategy(strategy, df)
                profitable_strategies_db.append((algo, strategy, profit))

    return results

# Main function to run the backtesting and ranking
def main():
    historical_data = pull_historical_data()

    # Use ThreadPoolExecutor to parallelize backtests
    with ThreadPoolExecutor(max_workers=4) as executor:
        for symbol, data in historical_data.items():
            for category, params in strategy_categories.items():
                executor.submit(run_algorithm_backtests, symbol, category, params)

    # Print top profitable strategies
    print("\nTop Profitable Strategies:")
    profitable_strategies_db.sort(key=lambda x: x[2], reverse=True)
    for i, (algo, strategy, profit) in enumerate(profitable_strategies_db[:10], 1):
        print(f"Rank {i}: Algorithm: {algo}, Strategy: {strategy}, Profit: {profit}")

if __name__ == "__main__":
    main()
