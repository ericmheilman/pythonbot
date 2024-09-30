#!/bin/bash

echo "Starting the Solana Trading Bot Setup..."

# Step 1: Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Installing Python 3..."
    sudo apt update
    sudo apt install python3 -y
else
    echo "Python 3 is already installed."
fi

# Step 2: Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip is not installed. Installing pip..."
    sudo apt install python3-pip -y
else
    echo "pip is already installed."
fi

# Step 3: Install virtualenv and set up a virtual environment
echo "Installing virtualenv..."
pip3 install virtualenv
echo "Creating virtual environment..."
virtualenv trading_bot_env
source trading_bot_env/bin/activate

# Step 4: Install necessary Python libraries
echo "Installing required Python packages..."
pip install ccxt ta-lib pandas numpy matplotlib pyalgotrade backtrader

# Step 5: Download and install TA-Lib (required for technical indicators)
echo "Downloading and installing TA-Lib..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..

# Step 6: Create the Solana Trading Bot template in bot.py
echo "Creating the Solana Trading Bot template..."
cat <<EOT >> bot.py
import ccxt
import pandas as pd
import pandas_ta as ta
import backtrader as bt

# Step 1: Fetch real-time data from Binance (or other exchange)
exchange = ccxt.binance()
symbol = 'SOL/USDT'

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Step 2: Calculate technical indicators
def calculate_indicators(df):
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['close'], length=20, std=2)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
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

# Step 4: Backtesting logic using Backtrader
class SolanaStrategy(bt.Strategy):
    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(period=20, devfactor=2)
        self.macd = bt.indicators.MACD()
        self.rsi = bt.indicators.RSI_Safe()
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()

    def next(self):
        if self.data.close[0] > self.bbands.lines.top[0]:  # Long signal
            self.buy()
        if self.data.close[0] < self.bbands.lines.bot[0]:  # Short signal
            self.sell()

# Step 5: Running the strategy
def run_backtest(df):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SolanaStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()
    cerebro.plot()

if __name__ == "__main__":
    df = fetch_data()
    df = calculate_indicators(df)
    signals = check_trading_signals(df)
    print(signals)
    run_backtest(df)
EOT

echo "Bot setup complete. To activate the virtual environment, use the command: 'source trading_bot_env/bin/activate'. Then, you can run your bot by executing 'python bot.py'."

