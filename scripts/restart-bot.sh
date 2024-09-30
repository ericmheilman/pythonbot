#!/bin/bash

# Define the name of your bot process
BOT_NAME="bot.py"

# Define the path to the Python interpreter inside the virtual environment
VENV_PYTHON="/Users/ericheilman/python-bot/trading_bot_env/bin/python3"

# Step 1: Check if the bot is running
echo "Checking if $BOT_NAME is running..."
PID=$(pgrep -f $BOT_NAME)

if [ -z "$PID" ]; then
  echo "$BOT_NAME is not running."
else
  echo "$BOT_NAME is running with PID: $PID. Stopping the bot..."
  kill $PID
fi

# Step 2: Ensure the virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
  echo "Virtual environment not found at $VENV_PYTHON"
  exit 1
fi

# Step 3: Run the bot using the virtual environment's Python
echo "Restarting $BOT_NAME..."
nohup $VENV_PYTHON bot.py > bot_output.log 2>&1 &

# Step 4: Confirm the bot is running
sleep 2
NEW_PID=$(pgrep -f $BOT_NAME)

if [ -z "$NEW_PID" ]; then
  echo "Failed to restart $BOT_NAME."
else
  echo "$BOT_NAME restarted successfully with PID: $NEW_PID."
fi

