import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt

# Step 1: Fetch real-time data from Binance (or another exchange)
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert the timestamp to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set the timestamp as the DataFrame index (which is required by Backtrader)
    df.set_index('timestamp', inplace=True)
    
    return df

# Step 2: Calculate technical indicators
def calculate_indicators(df):
    bb = ta.bbands(df['close'], length=20, std=2)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd, bb], axis=1)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

# Step 3: Define the Solana trading strategy
class SolanaStrategy(bt.Strategy):
    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(period=20, devfactor=2)
        self.macd = bt.indicators.MACD()
        self.rsi = bt.indicators.RSI_Safe()
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()
        self.order = None  # Track existing orders

    def next(self):
       if self.order:  # Check if there's already an active order
           return
    
       print(f"Price: {self.data.close[0]}, BB Upper: {self.bbands.lines.top[0]}, BB Lower: {self.bbands.lines.bot[0]}")
       print(f"RSI: {self.rsi[0]}, MACD Diff: {self.macd.macd[0] - self.macd.signal[0]}, ADX: {self.adx[0]}, ATR: {self.atr[0]}")

       # Long position logic
       if self.data.close[0] > self.bbands.lines.top[0]:
           wick_size = self.data.high[0] - self.data.close[0]
           print(f"Long Wick Size: {wick_size}, ATR: {self.atr[0]}")
           if wick_size >= self.atr[0]:
               macd_diff_now = self.macd.macd[0] - self.macd.signal[0]
               if macd_diff_now > 0 and self.rsi[0] < 70 and self.adx[0] > 25:
                   print("Entering Long Position")
                   self.order = self.buy()

       # Short position logic
       if self.data.close[0] < self.bbands.lines.bot[0]:
           wick_size = self.data.close[0] - self.data.low[0]
           print(f"Short Wick Size: {wick_size}, ATR: {self.atr[0]}")
           if wick_size >= self.atr[0]:
               macd_diff_now = self.macd.macd[0] - self.macd.signal[0]
               if macd_diff_now < 0 and self.rsi[0] > 30 and self.adx[0] > 25:
                   print("Entering Short Position")
                   self.order = self.sell()


import backtrader.analyzers as btanalyzers

# Step 4: Backtesting logic
def run_backtest(df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    
    cerebro.adddata(data)
    cerebro.addstrategy(SolanaStrategy)  # Use your custom strategy
    
    cerebro.broker.set_cash(10000)  # Set initial cash balance for backtest
    cerebro.broker.setcommission(commission=0.001)  # Set commission
    
    # Add analyzers for profitability metrics
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    
    # Run the backtest and get the results
    results = cerebro.run()
    
    # Get and print final portfolio value
    final_portfolio_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % final_portfolio_value)
    
    # Get analyzers
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
    
    # Plot the results
    cerebro.plot()

# Main entry point
if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)
    run_backtest(df)

# Main entry point
if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)
    run_backtest(df)

