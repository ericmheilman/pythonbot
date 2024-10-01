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

# Backtest a strategy using Backtrader with multi-timeframe data
def backtest_strategy(strategy, df, symbol):
    cerebro = bt.Cerebro()

    # Adding slippage and commission settings to make backtest more realistic
    cerebro.broker.set_slippage_perc(0.001)  # 0.1% slippage
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission per trade

    class TestStrategy(bt.Strategy):
        params = (
            ('macd_fast', 12),
            ('macd_slow', 26),
            ('macd_signal', 9),
            ('rsi_overbought', 70),
            ('rsi_oversold', 30),
            ('adx_threshold', 20),
            ('atr_period', 14),
            ('sma_period', 50),
            ('bollinger_period', 20),
            ('bollinger_dev', 2),
        )

        def __init__(self):
            # Indicators: MACD, RSI, ADX, SMA, ATR, Bollinger Bands, OBV
            self.macd = bt.indicators.MACD(
                period_me1=self.params.macd_fast,
                period_me2=self.params.macd_slow,
                period_signal=self.params.macd_signal
            )
            self.rsi = bt.indicators.RSI_Safe(period=14)
            self.adx = bt.indicators.ADX(period=14)
            self.atr = bt.indicators.ATR(period=self.params.atr_period)
            self.sma = bt.indicators.SimpleMovingAverage(period=self.params.sma_period)
            self.bollinger = bt.indicators.BollingerBands(period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
            self.obv = bt.indicators.OnBalanceVolume()

        def next(self):
            # Decision based on indicators
            if not self.position:  # Not already in a position
                if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
                    if self.data.close[0] > self.sma[0]:  # Price above SMA for a buy signal
                        self.buy()
                        logger.debug(f"Buy at {self.data.close[0]} on {self.data.datetime.datetime(0)}")
            elif self.position:  # Already in a position
                if self.rsi[0] > self.params.rsi_overbought or self.data.close[0] < self.sma[0]:  # Exit conditions
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
def setup_ga(symbol, category):
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_macd_fast", random.randint, 5, 15)  # Slightly wider range for fast MACD
    toolbox.register("attr_macd_slow", random.randint, 16, 30)  # Wider range for slow MACD
    toolbox.register("attr_macd_signal", random.randint, 5, 10)
    toolbox.register("attr_rsi_overbought", random.randint, 65, 80)
    toolbox.register("attr_rsi_oversold", random.randint, 20, 35)
    toolbox.register("attr_adx_threshold", random.randint, 15, 30)
    toolbox.register("attr_atr_period", random.randint, 10, 20)
    toolbox.register("attr_sma_period", random.randint, 40, 100)
    toolbox.register("attr_bollinger_period", random.randint, 15, 30)
    toolbox.register("attr_bollinger_dev", random.uniform, 1.5, 3.0)
    
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_macd_fast, toolbox.attr_macd_slow, toolbox.attr_macd_signal, 
                      toolbox.attr_rsi_overbought, toolbox.attr_rsi_oversold, toolbox.attr_adx_threshold,
                      toolbox.attr_atr_period, toolbox.attr_sma_period, toolbox.attr_bollinger_period, toolbox.attr_bollinger_dev), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate_ga_strategy(ind, symbol, category))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)  # Increased mutation probability for exploration
    toolbox.register("select", tools.selTournament, tournsize=5)  # Increased tournament size

    return toolbox

# Evaluate strategy generated by GA
def evaluate_ga_strategy(individual, symbol, category):
    strategy = {
        'macd_fast': max(1, individual[0]),
        'macd_slow': max(individual[0] + 1, individual[1]),
        'macd_signal': max(1, individual[2]),
        'rsi_overbought': individual[3],
        'rsi_oversold': individual[4],
        'adx_threshold': individual[5],
        'atr_period': individual[6],
        'sma_period': individual[7],
        'bollinger_period': individual[8],
        'bollinger_dev': individual[9],
    }

    df = fetch_multi_timeframe_data(symbol=symbol, timeframes=strategy_categories[category]['timeframes'])
    profit = backtest_strategy(strategy, df, symbol)
    return profit,

# Test each category, rank, and improve the best strategy
def test_and_rank_strategies():
    final_results = []
    
    for category, params in strategy_categories.items():
        logger.info(f"Testing {category} strategy...")
        symbol = 'solana'  # Replace with desired asset pair like SOL/USDT
        toolbox = setup_ga(symbol, category)
        population = toolbox.population(n=50)

        # Run Genetic Algorithm on this strategy
        algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.3, ngen=50, verbose=False)

        # Rank and get the best-performing strategy
        best_individual = max(population, key=lambda ind: ind.fitness.values[0])
        profit, strategy = evaluate_ga_strategy(best_individual, symbol, category)
        
        final_results.append((category, profit, strategy))
        profitable_strategies_db.append((category, best_individual, profit))

    # After one test cycle, rank the results
    final_results.sort(key=lambda x: x[1], reverse=True)
    logger.info("Rankings after first attempt:")
    for i, (category, profit, strategy) in enumerate(final_results, 1):
        logger.info(f"Rank {i}: {category} | Profit: {profit} | Strategy: {strategy}")

    # Now optimize the best strategy
    best_category, best_profit, best_strategy = final_results[0]
    logger.info(f"Optimizing the best strategy: {best_category}")

    toolbox = setup_ga('solana', best_category)
    population = toolbox.population(n=100)  # Increase population size for refinement
    algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.3, ngen=100, verbose=False)

    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    optimized_profit, optimized_strategy = evaluate_ga_strategy(best_individual, 'solana', best_category)
    
    # Final results after optimization
    logger.info("\nFinal Optimized Strategy:")
    logger.info(f"Best Category: {best_category}")
    logger.info(f"Optimized Profit: {optimized_profit}")
    logger.info(f"Optimized Strategy: {optimized_strategy}")

    return final_results, (best_category, optimized_profit, optimized_strategy)

# Main function
def main():
    first_attempt_rankings, final_optimized_strategy = test_and_rank_strategies()

if __name__ == "__main__":
    main()
