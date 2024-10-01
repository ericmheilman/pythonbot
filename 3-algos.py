import random
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Example trading categories with real-timeframes
strategy_categories = {
    "Scalping": {"timeframe": "1m", "history_limit": 5000},
    "Swing Trading": {"timeframe": "1h", "history_limit": 1000},
    "Position Trading": {"timeframe": "1d", "history_limit": 365},
    "Day Trading": {"timeframe": "15m", "history_limit": 2000},
}

def fetch_data(symbol, timeframe, limit):
    # Map trading timeframes to Pandas frequency strings
    timeframe_map = {
        '1m': 'min',   # Minutes
        '15m': '15min', # 15 Minutes
        '1h': 'h',     # Hourly
        '1d': 'D'      # Daily
    }
    
    # Make sure the timeframe is mapped correctly
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Use the mapped frequency string for the Pandas date_range
    freq = timeframe_map[timeframe]
    
    # Generate a range of timestamps with the appropriate frequency
    dates = pd.date_range(start="2023-01-01", periods=limit, freq=freq)
    
    # Fake price data (this should be replaced with actual API fetching)
    prices = np.random.normal(loc=20, scale=5, size=limit)
    
    # Create a DataFrame with the generated dates and price data
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": prices, "low": prices, "close": prices})
    df.set_index("timestamp", inplace=True)
    
    return df



# Backtest function for strategy profitability
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
        # Using self.params to access the strategy parameters
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
        elif self.rsi[0] > self.params.rsi_overbought and self.adx[0] > self.params.adx_threshold:
            self.sell()


    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()

    final_value = cerebro.broker.getvalue()
    return final_value - 10000  # Return profit

# Train and test AI models on real trading data (after backtesting)
def train_ai_models(X, y):
    models = {
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500),
        "SVM": SVR(),
        "Random Forest": RandomForestRegressor(),
    }
    
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results[model_name] = score
    
    return results

# Run the process for each trading category
def run_category_backtests():
    all_results = []

    # Loop over trading categories
    for category, params in strategy_categories.items():
        print(f"\nRunning backtests for {category}...")

        # Fetch price data for the given category
        df = fetch_data(symbol='SOL/USDT', timeframe=params['timeframe'], limit=params['history_limit'])

        # Create a random initial strategy for testing
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

        # Backtest the strategy and get profit
        profit = backtest_strategy(strategy, df)
        print(f"Profit for {category}: {profit}")
        
        # Store the results
        all_results.append((strategy, profit))

    return all_results

# Main function to test all trading categories
def main():
    # Run backtests on all categories and collect results
    all_results = run_category_backtests()

    # Convert results into DataFrame for AI model training
    df = pd.DataFrame([r[0] for r in all_results])  # Convert strategies to DataFrame
    df['profit'] = [r[1] for r in all_results]  # Add profit as target variable

    X = df.drop(columns=['profit'])  # Features
    y = df['profit']  # Target (profit)

    # Train and test AI models on the data
    ai_results = train_ai_models(X, y)

    # Print AI model results
    print("\nAI Model Performance:")
    for model_name, score in ai_results.items():
        print(f"{model_name}: {score}")

if __name__ == "__main__":
    main()
