import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
from sklearn.model_selection import ParameterGrid

# Step 1: Fetch real-time data from Binance (or other exchange)
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    """Fetch historical OHLCV data from Binance"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Ensure it's in datetime format
    df.set_index('timestamp', inplace=True)  # Backtrader expects the index to be datetime
    return df


def calculate_indicators(df):
    """Calculate key technical indicators: Bollinger Bands, MACD, RSI, ADX, ATR"""
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)  # Returns a DataFrame
    df['bb_upper'] = bb['BBU_20_2.0']  # Upper band
    df['bb_lower'] = bb['BBL_20_2.0']  # Lower band

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # RSI, ADX, ATR
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df


class SolanaFuturesStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_overbought', 70),  # RSI overbought threshold
        ('rsi_oversold', 30),    # RSI oversold threshold
        ('adx_threshold', 25),   # ADX threshold
        ('atr_multiplier', 2),   # ATR multiplier for stop-loss
        ('target_profit', 2),    # Risk-to-reward ratio for take-profit
    )

    def __init__(self):
        # Indicators
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
        # Example Long Entry Signal
        if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
            self.buy()
            stop_loss = self.data.close[0] - (self.atr[0] * self.params.atr_multiplier)
            take_profit = self.data.close[0] + (self.atr[0] * self.params.target_profit)
            self.sell(exectype=bt.Order.Stop, price=stop_loss)
            self.sell(exectype=bt.Order.Limit, price=take_profit)

        # Example Short Entry Signal
        elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.sell()
            stop_loss = self.data.close[0] + (self.atr[0] * self.params.atr_multiplier)
            take_profit = self.data.close[0] - (self.atr[0] * self.params.target_profit)
            self.buy(exectype=bt.Order.Stop, price=stop_loss)
            self.buy(exectype=bt.Order.Limit, price=take_profit)


# Backtesting
def run_backtest(df, strategy_params=None):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)
    cerebro.addstrategy(SolanaFuturesStrategy, **strategy_params)
    cerebro.broker.set_cash(10000)  # Starting capital

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run the strategy
    results = cerebro.run()

    # Get final portfolio value
    final_portfolio_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % final_portfolio_value)

    # Extract analyzers for performance metrics
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # Print the performance analysis
    print("\nTrade Analysis Results:")
    print(trade_analyzer)

    print("\nSharpe Ratio:")
    print(sharpe_ratio)

    print("\nDrawdown Analysis:")
    print(drawdown)

    # Plot the strategy results
    cerebro.plot()

# AI Optimization Capability
def ai_optimize_strategy(df, param_grid):
    """Optimize strategy parameters using AI/Hyperopt"""
    best_params = None
    best_profit = -float('inf')

    # Grid search for parameter optimization (can be replaced with AI search algorithms)
    for params in ParameterGrid(param_grid):
        print(f"Testing strategy with params: {params}")
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(SolanaFuturesStrategy, **params)
        cerebro.broker.set_cash(10000)
        results = cerebro.run()
        final_portfolio_value = cerebro.broker.getvalue()

        # Track the best-performing parameters
        if final_portfolio_value > best_profit:
            best_profit = final_portfolio_value
            best_params = params

    print(f"Best strategy params: {best_params}, Profit: {best_profit}")
    return best_params

if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)

    # Define parameter grid for optimization (this can be expanded for AI optimization)
    param_grid = {
        'macd_fast': [8, 12],
        'macd_slow': [26, 30],
        'macd_signal': [9],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
        'adx_threshold': [20, 25, 30],
        'atr_multiplier': [1.5, 2, 2.5],
        'target_profit': [1.5, 2, 2.5]
    }

    # Run AI Optimization (this can be improved with AI techniques like hyperopt, reinforcement learning, etc.)
    best_params = ai_optimize_strategy(df, param_grid)
    
    # Run backtest with the best parameters
    run_backtest(df, strategy_params=best_params)
import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime
from sklearn.model_selection import ParameterGrid

# Step 1: Fetch real-time data from Binance (or other exchange)
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    """Fetch historical OHLCV data from Binance"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Ensure it's in datetime format
    df.set_index('timestamp', inplace=True)  # Backtrader expects the index to be datetime
    return df


def calculate_indicators(df):
    """Calculate key technical indicators: Bollinger Bands, MACD, RSI, ADX, ATR"""
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)  # Returns a DataFrame
    df['bb_upper'] = bb['BBU_20_2.0']  # Upper band
    df['bb_lower'] = bb['BBL_20_2.0']  # Lower band

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # RSI, ADX, ATR
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df


class SolanaFuturesStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_overbought', 70),  # RSI overbought threshold
        ('rsi_oversold', 30),    # RSI oversold threshold
        ('adx_threshold', 25),   # ADX threshold
        ('atr_multiplier', 2),   # ATR multiplier for stop-loss
        ('target_profit', 2),    # Risk-to-reward ratio for take-profit
    )

    def __init__(self):
        # Indicators
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
        # Example Long Entry Signal
        if self.rsi[0] < self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
            self.buy()
            stop_loss = self.data.close[0] - (self.atr[0] * self.params.atr_multiplier)
            take_profit = self.data.close[0] + (self.atr[0] * self.params.target_profit)
            self.sell(exectype=bt.Order.Stop, price=stop_loss)
            self.sell(exectype=bt.Order.Limit, price=take_profit)

        # Example Short Entry Signal
        elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.sell()
            stop_loss = self.data.close[0] + (self.atr[0] * self.params.atr_multiplier)
            take_profit = self.data.close[0] - (self.atr[0] * self.params.target_profit)
            self.buy(exectype=bt.Order.Stop, price=stop_loss)
            self.buy(exectype=bt.Order.Limit, price=take_profit)


# Backtesting
def run_backtest(df, strategy_params=None):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)
    cerebro.addstrategy(SolanaFuturesStrategy, **strategy_params)
    cerebro.broker.set_cash(10000)  # Starting capital

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run the strategy
    results = cerebro.run()

    # Get final portfolio value
    final_portfolio_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % final_portfolio_value)

    # Extract analyzers for performance metrics
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # Print the performance analysis
    print("\nTrade Analysis Results:")
    print(trade_analyzer)

    print("\nSharpe Ratio:")
    print(sharpe_ratio)

    print("\nDrawdown Analysis:")
    print(drawdown)

    # Plot the strategy results
    cerebro.plot()

# AI Optimization Capability
def ai_optimize_strategy(df, param_grid):
    """Optimize strategy parameters using AI/Hyperopt"""
    best_params = None
    best_profit = -float('inf')

    # Grid search for parameter optimization (can be replaced with AI search algorithms)
    for params in ParameterGrid(param_grid):
        print(f"Testing strategy with params: {params}")
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(SolanaFuturesStrategy, **params)
        cerebro.broker.set_cash(10000)
        results = cerebro.run()
        final_portfolio_value = cerebro.broker.getvalue()

        # Track the best-performing parameters
        if final_portfolio_value > best_profit:
            best_profit = final_portfolio_value
            best_params = params

    print(f"Best strategy params: {best_params}, Profit: {best_profit}")
    return best_params

if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)

    # Define parameter grid for optimization (this can be expanded for AI optimization)
    param_grid = {
        'macd_fast': [8, 12],
        'macd_slow': [26, 30],
        'macd_signal': [9],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
        'adx_threshold': [20, 25, 30],
        'atr_multiplier': [1.5, 2, 2.5],
        'target_profit': [1.5, 2, 2.5]
    }

    # Run AI Optimization (this can be improved with AI techniques like hyperopt, reinforcement learning, etc.)
    best_params = ai_optimize_strategy(df, param_grid)
    
    # Run backtest with the best parameters
    run_backtest(df, strategy_params=best_params)
