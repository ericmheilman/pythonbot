import ccxt
import random
import backtrader as bt
import pandas as pd
import logging
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# File handler for verbose logging
file_handler = logging.FileHandler('backtest.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Adjusted strategy categories with slightly increased data limits for better model training
COMPLEXITY_DIVIDER = 20
strategy_categories = {
    "Scalping": {"timeframe": "1m", "history_limit": 500},  # Increased data size
    "Swing Trading": {"timeframe": "1h", "history_limit": 500},
    "Position Trading": {"timeframe": "1d", "history_limit": 500},
    "Day Trading": {"timeframe": "15m", "history_limit": 500},
}

# Placeholder to store profitable strategies
profitable_strategies_db = []

# Fetch data directly without using a map
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.binance()

    if not '/' in symbol:
        raise ValueError(f"Unsupported symbol format: {symbol}")

    logger.debug(f"Fetching {limit} bars for {symbol} on {timeframe} timeframe")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Backtest a strategy using Backtrader
def backtest_strategy(strategy, df, symbol):
    cerebro = bt.Cerebro()

    # Adding slippage and commission settings to make backtest more realistic
    cerebro.broker.set_slippage_perc(0.001)  # 0.1% slippage
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission per trade

    class TestStrategy(bt.Strategy):
        params = (
            ('macd_fast', 3),  # Shorter MACD periods
            ('macd_slow', 9),
            ('macd_signal', 3),
            ('rsi_overbought', 65),  # Lower overbought threshold for short-term trading
            ('rsi_oversold', 35),  # Lower oversold threshold for short-term trading
            ('adx_threshold', 20),  # Lower ADX threshold
        )

        def __init__(self):
            self.macd = bt.indicators.MACD(
                period_me1=max(1, strategy['macd_fast']),
                period_me2=max(strategy['macd_fast'] + 1, strategy['macd_slow']),
                period_signal=max(1, strategy['macd_signal'])
            )
            self.rsi = bt.indicators.RSI_Safe()
            self.adx = bt.indicators.ADX()

        def next(self):
            if not self.position:  # Not already in a position
                if self.rsi[0] < strategy['rsi_oversold'] and self.adx[0] > strategy['adx_threshold']:
                    self.buy()
                    logger.debug(f"Buy at {self.data.close[0]} on {self.data.datetime.datetime(0)}")
            elif self.position:  # Already in a position
                if self.rsi[0] > strategy['rsi_overbought'] and self.adx[0] > strategy['adx_threshold']:
                    self.sell()
                    logger.debug(f"Sell at {self.data.close[0]} on {self.data.datetime.datetime(0)}")

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()

    profit = cerebro.broker.getvalue() - 10000
    logger.info(f"Profit for {symbol}: {profit}")
    return profit

# Define the Genetic Algorithm (GA) setup
def setup_ga():
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_macd_fast", random.randint, 2, 5)  # Narrower range for fast MACD
    toolbox.register("attr_macd_slow", random.randint, 6, 12)  # Narrower range for slow MACD
    toolbox.register("attr_macd_signal", random.randint, 3, 6)
    toolbox.register("attr_rsi_overbought", random.randint, 60, 75)
    toolbox.register("attr_rsi_oversold", random.randint, 25, 40)
    toolbox.register("attr_adx_threshold", random.randint, 15, 30)
    toolbox.register("attr_target_profit", random.uniform, 1.5, 3.0)

    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_macd_fast, toolbox.attr_macd_slow, toolbox.attr_macd_signal, 
                      toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold, toolbox.attr_adx_threshold, 
                      toolbox.attr_target_profit), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_ga_strategy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)  # Increased mutation probability for exploration
    toolbox.register("select", tools.selTournament, tournsize=5)  # Increased tournament size

    return toolbox

# Evaluate strategy generated by GA
def evaluate_ga_strategy(individual):
    strategy = {
        'macd_fast': max(1, individual[0]),
        'macd_slow': max(individual[0] + 1, individual[1]),
        'macd_signal': max(1, individual[2]),
        'rsi_overbought': individual[3],
        'rsi_oversold': individual[4],
        'adx_threshold': individual[5],
        'target_profit': individual[6],
    }

    df = fetch_data(symbol='SOL/USDT', timeframe='1m', limit=500)  # Increased limit for more meaningful results
    profit = backtest_strategy(strategy, df, 'SOL/USDT')
    return profit,

# Pull historical data for SOL, ETH, and BTC
def pull_historical_data():
    sol_data = fetch_data('SOL/USDT', '1m', 500)  # Increased limit
    eth_data = fetch_data('ETH/USDT', '1m', 500)  # Increased limit
    btc_data = fetch_data('BTC/USDT', '1m', 500)  # Increased limit
    return {'SOL/USDT': sol_data, 'ETH/USDT': eth_data, 'BTC/USDT': btc_data}

# Run Genetic Algorithm (GA)
def run_algorithm_backtests(symbol, category, params):
    df = fetch_data(symbol=symbol, timeframe=params['timeframe'], limit=params['history_limit'])

    logger.info(f"Running Genetic Algorithm for {symbol} on {category}")
    toolbox = setup_ga()
    population = toolbox.population(n=50)  # Increased population size
    algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.3, ngen=50, verbose=False)  # Increased generations

    ranked_strategies = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    for strategy in ranked_strategies[:3]:  # Top 3 strategies
        profit = evaluate_ga_strategy(strategy)
        profitable_strategies_db.append(('Genetic Algorithm', strategy, profit))

# Main function to run the backtesting and ranking
def main():
    historical_data = pull_historical_data()

    # Use ThreadPoolExecutor to parallelize backtests
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        total_tests = len(historical_data) * len(strategy_categories)  # Only running GA
        completed_tests = 0

        for symbol, data in historical_data.items():
            for category, params in strategy_categories.items():
                futures.append(executor.submit(run_algorithm_backtests, symbol, category, params))

        for future in futures:
            future.result()
            completed_tests += 1
            logger.info(f"Progress: {completed_tests}/{total_tests} tests completed.")

    # Print top profitable strategies
    logger.info("\nTop Profitable Strategies:")
    profitable_strategies_db.sort(key=lambda x: x[2], reverse=True)
    for i, (algo, strategy, profit) in enumerate(profitable_strategies_db[:10], 1):
        logger.info(f"Rank {i}: Algorithm: {algo}, Strategy: {strategy}, Profit: {profit}")

if __name__ == "__main__":
    main()
