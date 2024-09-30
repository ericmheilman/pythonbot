import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt
import datetime

# Step 1: Fetch real-time data from Binance (or other exchange)
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Ensure it's in datetime format
    df.set_index('timestamp', inplace=True)  # Backtrader expects the index to be datetime
    return df

def calculate_indicators(df):
    bb = ta.bbands(df['close'], length=20, std=2)  # This returns a DataFrame
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    
    # Accessing the bands directly
    df['bb_upper'] = bb['BBU_20_2.0']  # Upper band
    df['bb_middle'] = bb['BBM_20_2.0']  # Middle band
    df['bb_lower'] = bb['BBL_20_2.0']  # Lower band
    
    df = pd.concat([df, macd], axis=1)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df

# Step 3: Trading logic based on strategy
def check_trading_signals(df):
    signals = []
    for i in range(1, len(df)):
        # Long position entry
        if df['close'][i] > df['bb_upper'][i]:
            wick_size = df['high'][i] - df['close'][i]
            if wick_size >= df['atr'][i]:
                macd_diff_now = df['MACD_12_26_9'][i] - df['MACDs_12_26_9'][i]
                macd_diff_prev = df['MACD_12_26_9'][i-1] - df['MACDs_12_26_9'][i-1]
                if macd_diff_now > macd_diff_prev and df['rsi'][i] < 70 and df['adx'][i] > 25:
                    signals.append((df['timestamp'][i], 'BUY', df['close'][i]))

        # Short position entry
        elif df['close'][i] < df['bb_lower'][i]:
            wick_size = df['close'][i] - df['low'][i]
            if wick_size >= df['atr'][i]:
                macd_diff_now = df['MACD_12_26_9'][i] - df['MACDs_12_26_9'][i]
                macd_diff_prev = df['MACD_12_26_9'][i-1] - df['MACDs_12_26_9'][i-1]
                if macd_diff_now < macd_diff_prev and df['rsi'][i] > 30 and df['adx'][i] > 25:
                    signals.append((df['timestamp'][i], 'SELL', df['close'][i]))

    return signals

class SolanaStrategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_overbought', 70),  # Overbought threshold for RSI
        ('rsi_oversold', 30),    # Oversold threshold for RSI
        ('adx_threshold', 25),   # ADX threshold
    )
    
    def __init__(self):
        # MACD indicator
        self.macd = bt.indicators.MACD(
            period_me1=self.params.macd_fast, 
            period_me2=self.params.macd_slow, 
            period_signal=self.params.macd_signal
        )
        
        # Bollinger Bands
        self.bbands = bt.indicators.BollingerBands(period=20, devfactor=2)
        
        # RSI and other indicators
        self.rsi = bt.indicators.RSI_Safe()
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()

    def next(self):
        # Buy signal
        if self.data.close[0] > self.bbands.lines.top[0] and self.rsi[0] < self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.buy()

        # Sell signal
        if self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] > self.params.rsi_oversold and self.adx[0] > self.params.adx_threshold:
            self.sell()

# Backtesting
def run_backtest(df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)
    cerebro.addstrategy(SolanaStrategy)
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

if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)
    signals = check_trading_signals(df)
    print(signals)
    run_backtest(df)
