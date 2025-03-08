import torch
import os
import json
import sys
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay 
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import argparse
matplotlib.use('Agg')

def gen_covariates(times, price_data, num_covariates, with_sentiment=False):
    """
    Creates additional features including technical indicators
    Args:
        times: DatetimeIndex for the time series
        price_data: DataFrame containing price data and possibly sentiment
        num_covariates: Number of features to generate
        with_sentiment: Boolean indicating if sentiment data is included
    Returns:
        ndarray containing all computed features
    """
    covariates = np.zeros((len(times), num_covariates))
    
    # Time-based features
    covariates[:, 0] = stats.zscore([t.weekday() for t in times])
    covariates[:, 1] = stats.zscore([t.month for t in times])
    
    # Price-based features
    covariates[:, 2] = stats.zscore(price_data['Close'].shift(5).values)
    covariates[:, 3] = stats.zscore(price_data['Volume'].shift(5).values)
    
    # Technical indicators
    intraday_return = (price_data['Close'] - price_data['Open']) / price_data['Open']
    covariates[:, 4] = stats.zscore(intraday_return.shift(5).values)
    
    ma5 = price_data['Close'].rolling(window=5).mean()
    covariates[:, 5] = stats.zscore((price_data['Close'] - ma5).values)
    
    exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    covariates[:, 6] = stats.zscore(macd.shift(2).values)
    
    volatility = (price_data['High'] - price_data['Low']) / price_data['Close']
    covariates[:, 7] = stats.zscore(volatility.values)
    
    # Add sentiment feature if available
    if with_sentiment and 'Sentiment_Score' in price_data.columns:
        covariates[:, 8] = stats.zscore(price_data['Sentiment_Score'].shift(5).values)
    
    return np.nan_to_num(covariates)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stock Price Prediction with DeepAR")
    parser.add_argument('ticker', type=str, help="Stock ticker symbol (e.g., AAPL, MSFT, GOOG)")
    parser.add_argument('--with_sentiment', action='store_true', help="Include sentiment analysis in feature set")
    parser.add_argument('--forecast_start', type=str, default="2025-02-18", help="Start date for forecast")
    parser.add_argument('--end_date', type=str, default="2025-02-24", help="End date for evaluation")
    parser.add_argument('--train_window', type=int, default=30, help="Training window size")
    parser.add_argument('--future_steps', type=int, default=5, help="Number of days to predict")
    
    args = parser.parse_args()
    
    # Determine number of covariates based on whether sentiment is included
    num_covariates = 9 if args.with_sentiment else 8
    
    # Setup paths
    capstone_dir = os.path.join(os.path.expanduser("~"), "CAPSTONE-stockreturn")
    model_dir = os.path.join(capstone_dir, "DeepAR", "experiments", f"{args.ticker.lower()}_base_model")
    
    # Determine data path based on whether sentiment is included
    if args.with_sentiment:
        data_path = os.path.join(capstone_dir, "data", "stock", f"{args.ticker.lower()}_stock_wsenti.csv")
    else:
        data_path = os.path.join(capstone_dir, "data", "stock", f"{args.ticker.lower()}_stock_data.csv")
    
    # Load the dataset
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index("Date", inplace=True)
    
    # Parameters for prediction
    forecast_start = args.forecast_start
    end_date = args.end_date
    train_window = args.train_window
    future_steps = args.future_steps
    
    # Add path for model imports
    sys.path.append(os.path.join(capstone_dir, "DeepAR"))
    
    # Import after path setup
    import utils
    import model.net as net
    
    # Initialize parameters
    params = utils.Params(os.path.join(model_dir, "params.json"))
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the checkpoint files to process
    checkpoint_files = [f"epoch_{i}.pth.tar" for i in range(15)] + ["best.pth.tar"]
    
    # Loop through each checkpoint
    for checkpoint_file in checkpoint_files:
        print(f"Processing checkpoint: {checkpoint_file}")
        
        # Extract epoch number or 'best' for filename
        if checkpoint_file == "best.pth.tar":
            epoch_label = "best"
        else:
            epoch_label = checkpoint_file.split("_")[1].split(".")[0]
        
        # Load the model for this checkpoint
        model = net.Net(params).to(params.device)
        utils.load_checkpoint(os.path.join(model_dir, checkpoint_file), model)
        model.eval()
        
        # Get the training data
        last_30_days = data.loc[:forecast_start].iloc[-train_window:]
        last_30_days.fillna(method='ffill', inplace=True)
        
        # Select features based on whether sentiment is included
        if args.with_sentiment:
            price_data = last_30_days[['High', 'Low', 'Open', 'Close', 'Volume', 'Sentiment_Score', 'Daily Return']]
        else:
            price_data = last_30_days[['High', 'Low', 'Open', 'Close', 'Volume', 'Daily Return']]
        
        # Generate covariates
        covariates = gen_covariates(last_30_days.index, price_data, num_covariates, args.with_sentiment)
        
        # Prepare initial input tensor
        x_input = np.zeros((1, train_window, 1 + num_covariates), dtype='float32')
        x_input[0, 1:, 0] = last_30_days['Daily Return'].values[1:]
        x_input[0, :, 1:1 + num_covariates] = covariates[-train_window:, :]
        new_input_tensor = torch.tensor(x_input, dtype=torch.float32).permute(1, 0, 2).to(params.device)
        
        # Generate future trading days
        start_date = pd.to_datetime(forecast_start)
        future_trading_days = pd.date_range(start=start_date, periods=future_steps, freq=BDay())
        
        # Predict for trading days
        batch_size = new_input_tensor.shape[1]
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)
        idx = torch.zeros(1, batch_size, dtype=torch.long, device=params.device)
        predictions = []
        
        for _ in range(future_steps):
            mu, sigma, hidden, cell = model(new_input_tensor[-1].unsqueeze(0), idx, hidden, cell)
            next_value = mu.cpu().detach().numpy().squeeze()
            predictions.append(next_value)
            new_input = np.roll(new_input_tensor.cpu().numpy(), shift=-1, axis=0)
            new_input[-1, 0, 0] = next_value
            new_input_tensor = torch.tensor(new_input, dtype=torch.float32).to(params.device)
        
        # Convert returns to stock prices
        last_price = last_30_days['Close'].iloc[-1]
        predicted_prices = [last_price]
        for ret in predictions:
            next_price = predicted_prices[-1] * (1 + ret)
            predicted_prices.append(next_price)
        predicted_prices = predicted_prices[1:]  # Remove the initial price
        
        # Get actual data for comparison
        actual_data = data.reset_index()
        actual_data = actual_data[(actual_data["Date"] >= forecast_start) & (actual_data["Date"] <= end_date)]
        
        # Create the combined plot
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.plot(future_trading_days, predicted_prices, marker='o', linestyle='-', color='blue', label="Predicted Prices")
        
        # Plot actual prices
        plt.plot(actual_data["Date"], actual_data["Close"], marker='x', linestyle='--', color='green', label="Actual Prices")
        
        # Add vertical line at prediction start
        plt.axvline(x=pd.to_datetime(forecast_start), color='red', linestyle='--', label='Prediction Start')
        
        # Format the plot
        plt.title(f"{args.ticker.upper()} Stock: Predicted vs Actual Prices (Checkpoint: {checkpoint_file})")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # Calculate metrics
        actual_data_dates_str = [d.strftime('%Y-%m-%d') for d in actual_data["Date"]]
        future_days_str = [d.strftime('%Y-%m-%d') for d in future_trading_days]
        
        matching_actual = []
        matching_indices = []
        
        for i, pred_date_str in enumerate(future_days_str):
            if pred_date_str in actual_data_dates_str:
                idx = actual_data_dates_str.index(pred_date_str)
                matching_actual.append(actual_data["Close"].iloc[idx])
                matching_indices.append(i)
        
        matching_actual = np.array(matching_actual)
        predicted_prices_array = np.array(predicted_prices)
        
        # Basic metrics
        mae = np.mean(np.abs(predicted_prices_array[matching_indices] - matching_actual))
        mape = np.mean(np.abs((matching_actual - predicted_prices_array[matching_indices]) / matching_actual)) * 100
        rmse = np.sqrt(np.mean((predicted_prices_array[matching_indices] - matching_actual)**2))
        
        # Direction accuracy
        if len(matching_actual) > 1:  # Ensure we have at least 2 points for direction
            actual_direction = np.diff(matching_actual) > 0
            predicted_direction = np.diff(predicted_prices_array[matching_indices]) > 0
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            direction_accuracy = float('nan')
        
        # Short-term vs long-term accuracy
        if len(matching_actual) >= 2:
            short_term_mae = np.mean(np.abs(predicted_prices_array[matching_indices[:2]] - matching_actual[:2]))
        else:
            short_term_mae = float('nan')
            
        if len(matching_actual) >= 5:
            long_term_mae = np.mean(np.abs(predicted_prices_array[matching_indices[3:]] - matching_actual[3:]))
        else:
            long_term_mae = float('nan')
        
        # Add a text box with metrics to the plot
        metrics_text = (
            f"Metrics:\n"
            f"MAE: ${mae:.2f}\n"
            f"MAPE: {mape:.2f}%\n"
            f"RMSE: ${rmse:.2f}\n"
            f"Direction Accuracy: {direction_accuracy:.1f}%\n"
            f"Short-term MAE (1-2d): ${short_term_mae:.2f}\n"
            f"Long-term MAE (4-5d): ${long_term_mae:.2f}"
        )
        
        # Position the text box in the upper right corner with some padding
        plt.annotate(
            metrics_text,
            xy=(0.97, 0.97),
            xycoords='axes fraction',
            fontsize=9,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure with a unique name based on the checkpoint
        figures_dir = os.path.join(model_dir, "figures", f"{args.ticker.lower()}_compare_epochs")
        os.makedirs(figures_dir, exist_ok=True)
        plot_path = os.path.join(figures_dir, f"{args.ticker.lower()}_prediction_vs_actual_with_metrics_{epoch_label}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Saved plot for checkpoint {checkpoint_file} to {plot_path}")
    
    print("Completed processing all checkpoints!")

if __name__ == "__main__":
    main()