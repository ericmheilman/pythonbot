import random
import backtrader as bt
import pandas as pd
import logging
from pycoingecko import CoinGeckoAPI
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
import multiprocessing  # To get the number of available CPUs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# File handler for verbose logging
file_handler = logging.FileHandler('backtest.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Placeholder to store profitable strategies
profitable_strategies_db = []

# Fetch data using CoinGecko API for multiple timeframes
def fetch_multi_timeframe_data(symbol, vs_currency='usd', days_list=['1', '7', '30']):
    cg = CoinGeckoAPI()
    
    # Convert symbol to lowercase to match CoinGecko format
    symbol = symbol.lower()

    dfs = []
    for days in days_list:
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
        df['volume'] = 1  # CoinGecko doesn't provide volume, using a placeholder
        
        dfs.append(df)

    # Combine the dataframes from multiple timeframes into one DataFrame
    combined_df = pd.concat(dfs, axis=0).sort_index()
    return combined_df

# Backtest a strategy using Backtrader with multi-timeframe data
def backtest_strategy_with_multi_timeframes(strategy, df, symbol):
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
            # Decision-making process using multi-timeframe data context
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

    df = fetch_multi_timeframe_data(symbol='solana', days_list=['1', '7', '30'])  # Fetching multi-timeframe data
    profit = backtest_strategy_with_multi_timeframes(strategy, df, 'SOL/USDT')
    return profit,

# Main function to run the backtesting and ranking
def main():
    logger.info("Running Genetic Algorithm for SOL/USDT")

    # Maximize the number of workers using all available CPU cores
    max_workers = multiprocessing.cpu_count()

    toolbox = setup_ga()
    population = toolbox.population(n=50)  # Increased population size

    # Running Genetic Algorithm with parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_populations = [
            executor.submit(algorithms.eaSimple, population, toolbox, cxpb=0.6, mutpb=0.3, ngen=50, verbose=False)
            for _ in range(max_workers)
        ]

        for future in future_populations:
            future.result()

    # After parallel execution, get the top-ranked strategies
    ranked_strategies = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    for strategy in ranked_strategies[:3]:  # Top 3 strategies
        profit = evaluate_ga_strategy(strategy)
        profitable_strategies_db.append(('Genetic Algorithm', strategy, profit))

    # Log final results
    logger.info("\nTop Profitable Strategies:")
    profitable_strategies_db.sort(key=lambda x: x[2], reverse=True)
    for i, (algo, strategy, profit) in enumerate(profitable_strategies_db[:10], 1):
        logger.info(f"Rank {i}: Algorithm: {algo}, Strategy: {strategy}, Profit: {profit}")

if __name__ == "__main__":
    main()

