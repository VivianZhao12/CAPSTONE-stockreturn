import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os
import logging
from datetime import timedelta

def setup_logging(log_file):
    """Set up logging to both console and file"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
    
class FusionLayer:
    def __init__(self, model_type='rf', lookback_days=7):
        """
        Initialize the fusion layer
        
        Parameters:
        model_type: Fusion model type ('rf' for random forest, can be extended to other models)
        lookback_days: How many days of historical predictions to use as features
        """
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, daily_predictions_path, quarterly_data_path):
        """
        Load daily predictions and quarterly data
        """
        try:
            # Load data
            print(f"Attempting to load file: {daily_predictions_path}")
            self.daily_df = pd.read_csv(daily_predictions_path)
            print(f"Attempting to load file: {quarterly_data_path}")
            self.quarterly_df = pd.read_csv(quarterly_data_path)
            
            # Ensure date columns are in datetime format
            self.daily_df['Date'] = pd.to_datetime(self.daily_df['Date'])
            self.quarterly_df['fiscalDateEnding'] = pd.to_datetime(self.quarterly_df['fiscalDateEnding'])
            
            # Sort by date
            self.daily_df = self.daily_df.sort_values('Date')
            self.quarterly_df = self.quarterly_df.sort_values('fiscalDateEnding')
            
            print(f"Loaded {len(self.daily_df)} daily prediction records")
            print(f"Loaded {len(self.quarterly_df)} quarterly data records")
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            print(f"Current working directory: {os.getcwd()}")
            print("Please check if the file path is correct, or use absolute paths.")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
    def merge_quarterly_to_daily(self):
        """
        Map quarterly data to daily data
        Using forward fill method, each date uses the most recent available quarterly data
        """
        # Create date range from the earliest to the latest date in daily_df
        date_range = pd.date_range(start=self.daily_df['Date'].min(), 
                                   end=self.daily_df['Date'].max())
        date_df = pd.DataFrame({'Date': date_range})
        
        # Rename quarterly data for easier merging
        quarterly_renamed = self.quarterly_df.rename(columns={
            'fiscalDateEnding': 'Date'
        })
        
        # Merge quarterly data with date range
        merged_df = pd.merge_asof(date_df.sort_values('Date'), 
                                  quarterly_renamed.sort_values('Date'),
                                  on='Date', 
                                  direction='backward')
        
        # Merge quarterly data with daily predictions
        self.merged_df = pd.merge(self.daily_df, merged_df, on='Date', how='left')
        print(f"The merged data has {len(self.merged_df)} rows and {self.merged_df.shape[1]} columns")

    def create_features(self):
        """
        Create features for the fusion model
        Including:
        1. Original DeepAR predictions
        2. Rolling average prediction errors
        3. Recent macro/micro indicators
        4. Time features (seasonality, trends, etc.)
        5. Historical performance of predictions
        """
        df = self.merged_df.copy()
        
        # Create time features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
    
        # Add volatility features
        df['recent_volatility'] = df['prediction'].rolling(window=10).std()
        
        # Create rolling features
        for window in [3, 7, 14]:
            # Rolling average prediction
            df[f'prediction_rolling_{window}d'] = df['prediction'].rolling(window=window).mean()
            
            # Rolling average error
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df['error'].rolling(window=window).mean()
                df[f'abs_error_rolling_{window}d'] = df['abs_error'].rolling(window=window).mean()
            else:
                df[f'error_rolling_{window}d'] = 0
                df[f'abs_error_rolling_{window}d'] = 0
        
        # Calculate financial ratios using the correct column names
        # Check if columns exist
        revenue_col = 'INCOME_STATEMENT_totalRevenue'
        profit_col = 'INCOME_STATEMENT_grossProfit'
        cost_col = 'INCOME_STATEMENT_costOfRevenue'
        
        if revenue_col in df.columns and profit_col in df.columns:
            # Calculate year-over-year growth rates for quarterly revenue and profit
            df['revenue_yoy_growth'] = df.groupby(['quarter'])[revenue_col].pct_change(4)
            df['profit_yoy_growth'] = df.groupby(['quarter'])[profit_col].pct_change(4)
            
            # Calculate profit margin
            df['profit_margin'] = df[profit_col] / df[revenue_col]
            
            if cost_col in df.columns:
                # Try to extract relative magnitude of financial data, handling large numbers
                df['revenue_to_cost_ratio'] = df[revenue_col] / df[cost_col]
        
        # Add macroeconomic indicators (if they exist)
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL']
        for col in macro_cols:
            if col in df.columns:
                # Create some simple derived features
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_change'] = df[col].pct_change()
        
        # Add lagged features - predictions and errors from past days
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df['prediction'].shift(lag)
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df['error'].shift(lag)
            else:
                df[f'error_lag_{lag}'] = 0
        
        # Check for missing values
        print(f"Data has {len(df)} rows before feature engineering")
        
        # Print columns with highest missing value ratios
        na_ratio = df.isna().mean().sort_values(ascending=False)
        print("\nTop 10 columns with highest missing value ratios:")
        print(na_ratio.head(10))
        
        # First try to fill missing values in rolling features and lagged features
        for window in [3, 7, 14]:
            df[f'prediction_rolling_{window}d'] = df[f'prediction_rolling_{window}d'].fillna(df['prediction'])
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df[f'error_rolling_{window}d'].fillna(0)
                df[f'abs_error_rolling_{window}d'] = df[f'abs_error_rolling_{window}d'].fillna(0)
                
        # Fill lagged features
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df[f'prediction_lag_{lag}'].fillna(df['prediction'])
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df[f'error_lag_{lag}'].fillna(0)
        
        # Use forward fill for missing values in financial and macro data
        finance_cols = [col for col in df.columns if 'INCOME_STATEMENT_' in col or 'BALANCE_SHEET_' in col or 'CASH_FLOW_' in col]
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL', 'M2SL', 'M1SL', 'PPIACO', 'RTWEXBGS']
        fill_forward_cols = finance_cols + macro_cols
        
        for col in fill_forward_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Remove any rows that still contain NaN values
        self.feature_df = df.dropna()
        print(f"\nRemaining {len(self.feature_df)} rows of data after feature engineering")
        
        # If there isn't enough data, try more aggressive missing value handling
        if len(self.feature_df) < 30:  # Need at least 30 rows for training
            print("Insufficient data rows, trying more aggressive missing value handling...")
            # Calculate column missing ratios
            missing_ratio = df.isna().mean()
            
            # Drop columns with missing rate over 50%
            cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
            print(f"Dropping the following high-missing-rate columns: {cols_to_drop}")
            df_reduced = df.drop(columns=cols_to_drop)
            
            # Use 0 for numeric columns, mode for non-numeric columns
            numeric_cols = df_reduced.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df_reduced[col] = df_reduced[col].fillna(0)
                
            categorical_cols = df_reduced.select_dtypes(exclude=['float64', 'int64']).columns
            for col in categorical_cols:
                if col != 'Date':  # Don't process date columns
                    try:
                        mode_value = df_reduced[col].mode()[0]
                        df_reduced[col] = df_reduced[col].fillna(mode_value)
                    except:
                        pass
            
            self.feature_df = df_reduced
            print(f"Remaining {len(self.feature_df)} rows of data after aggressive processing")
            
        # If still not enough data, use very aggressive methods
        if len(self.feature_df) < 30:
            print("Data still insufficient, using minimum feature set...")
            min_features_df = df[['Date', 'prediction', 'day_of_week', 'month', 'quarter']].copy()
            
            if 'actual' in df.columns:
                min_features_df['actual'] = df['actual']
                
            # If there are other key features, add them
            for col in ['GDP', 'UNRATE']:
                if col in df.columns:
                    min_features_df[col] = df[col].fillna(df[col].median())
                    
            self.feature_df = min_features_df.dropna()
            print(f"Remaining {len(self.feature_df)} rows of data after minimum feature set")

    def train_model(self, test_size=0.2):
        """
        Train the fusion model
        """
        if len(self.feature_df) == 0:
            raise ValueError("Not enough data for training, please check data processing steps")
            
        # Ensure the dataset is large enough to split
        if len(self.feature_df) < 5:
            print("Warning: Dataset too small to split. Using all data for training.")
            test_size = 0
            
        # Define feature and target columns
        exclude_cols = [
            'Date', 'actual', 'error', 'abs_error', 'correct_direction', 
            'Ticker', 'fiscalDateEnding', 'observation_date'
        ]
        
        # Ensure 'actual' column exists
        if 'actual' not in self.feature_df.columns:
            raise ValueError("Missing 'actual' column in the data, cannot train the model")
            
        # Dynamically determine feature columns
        features = [col for col in self.feature_df.columns if col not in exclude_cols]
        
        X = self.feature_df[features]
        y = self.feature_df['actual']
        
        # Handle potential feature issues
        # Remove columns where all values are the same (no information)
        constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        if constant_cols:
            print(f"Removing the following constant columns: {constant_cols}")
            X = X.drop(columns=constant_cols)
            
        # Check if there are still features available
        if X.shape[1] == 0:
            raise ValueError("No features available after processing")
            
        # Split into training and test sets (if test size is greater than 0)
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X.iloc[:1], y.iloc[:1]  # Create a minimal test set
        
        # Standardize numeric features
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X_train[numeric_features])
        X_train[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        # Save the list of features used, for use in prediction
        self.used_features = X_train.columns.tolist()
        
        # Initialize and train model
        
        # Standardize numeric features
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X_train[numeric_features])
        X_train[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        # Initialize and train model
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Can add other model types
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        
        # Feature importance
        if self.model_type == 'rf':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature importance:")
            print(feature_importance.head(10))
        
        print(f"\nTraining set RMSE: {train_rmse:.6f}")
        print(f"Test set RMSE: {test_rmse:.6f}")
        print(f"Training set MAE: {train_mae:.6f}")
        print(f"Test set MAE: {test_mae:.6f}")
        
        # Save evaluation results
        self.evaluation = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
        }
        
        return self.evaluation

    def predict(self, new_data=None):
        """
        Use the trained model to make predictions, and fuse with DeepAR predictions
        If new_data is provided, use it, otherwise use the training data
        """
        # Store features used during training
        if not hasattr(self, 'used_features'):
            raise ValueError("Model not yet trained, cannot get used features")
            
        if new_data is not None:
            # Ensure all training features are available in the prediction data
            for feature in self.used_features:
                if feature not in new_data.columns:
                    print(f"Warning: Missing feature '{feature}', filling with 0")
                    new_data[feature] = 0
                    
            # Only use features that were used during training
            X = new_data[self.used_features]
            # Get original DeepAR predictions
            deepar_predictions = new_data['prediction'].values
        else:
            # Use training data
            X = self.feature_df[self.used_features]
            # Get original DeepAR predictions
            deepar_predictions = self.feature_df['prediction'].values
        
        # Standardize numeric features
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # Get base predictions from the fusion model
        base_predictions = self.model.predict(X)
        
        # Detect potential volatility
        if 'recent_volatility' in X.columns:
            volatility = X['recent_volatility'].values
            # Dynamically adjust weights - give DeepAR more weight during high volatility periods
            alpha = np.ones_like(base_predictions) * 0.6  # Default weight
            high_vol_mask = volatility > np.median(volatility)
            alpha[high_vol_mask] = 0.4  # Reduce fusion model weight during high volatility
        else:
            # Use fixed weight
            alpha = 0.6
        
        # Fuse predictions, preserving more of DeepAR's volatility
        final_predictions = alpha * base_predictions + (1 - alpha) * deepar_predictions
        
        return final_predictions

    def predict_future(self, future_dates_df, future_quarterly_data=None):
        """
        Predict future time periods
        
        Parameters:
        future_dates_df: DataFrame containing future dates and DeepAR predictions
        future_quarterly_data: Optional future quarterly data
        
        Returns:
        DataFrame with fusion predictions
        """
        print("Predicting future time periods...")
        
        # Merge future quarterly data (if available)
        if future_quarterly_data is not None:
            # Ensure date format is correct
            future_quarterly_data['fiscalDateEnding'] = pd.to_datetime(future_quarterly_data['fiscalDateEnding'])
            # Add to existing quarterly data
            combined_quarterly = pd.concat([self.quarterly_df, future_quarterly_data]).drop_duplicates()
            # Update quarterly data
            self.quarterly_df = combined_quarterly
        
        # Prepare data for future dates
        future_df = future_dates_df.copy()
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        
        # Get latest quarterly data and forward fill
        latest_quarterly = self.quarterly_df.sort_values('fiscalDateEnding').iloc[-1:].copy()
        
        # Predict financial data for the next quarter
        next_quarter_pred = self.predict_next_quarters(num_quarters=4)
        
        # Combine actual quarterly data and predicted quarterly data
        all_quarterly = pd.concat([self.quarterly_df, next_quarter_pred])
        
        # Map quarterly data to future dates
        quarterly_renamed = all_quarterly.rename(columns={'fiscalDateEnding': 'Date'})
        future_with_quarterly = pd.merge_asof(
            future_df.sort_values('Date'), 
            quarterly_renamed.sort_values('Date'),
            on='Date', 
            direction='backward'
        )
        
        # Create features
        future_feature_df = self.prepare_future_features(future_with_quarterly)
        
        # Ensure all needed features exist
        if not hasattr(self, 'used_features'):
            print("Warning: Model did not save the list of used features, attempting to use all available features")
            # Use all possible features
            exclude_cols = [
                'Date', 'actual', 'error', 'abs_error', 'correct_direction', 
                'Ticker', 'fiscalDateEnding', 'observation_date'
            ]
            features = [col for col in future_feature_df.columns if col not in exclude_cols]
        else:
            features = self.used_features
            
        # Ensure all needed features exist
        for feature in features:
            if feature not in future_feature_df.columns:
                print(f"Adding missing feature: {feature}")
                future_feature_df[feature] = 0
        
        # Use trained model to predict
        X = future_feature_df[features]
        
        # Standardize features
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # Predict
        future_predictions = self.model.predict(X)
        
        # Add prediction results
        result_df = future_feature_df.copy()
        result_df['fusion_prediction'] = future_predictions
        
        print(f"Prediction complete, {len(result_df)} predictions made")
        
        return result_df
    
    def prepare_future_features(self, future_df):
        """
        Prepare features for future data
        """
        df = future_df.copy()
        
        # Create time features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Create rolling features
        for window in [3, 7, 14]:
            # Rolling average prediction
            df[f'prediction_rolling_{window}d'] = df['prediction'].rolling(window=window).mean()
            
            # If historical data is available, calculate rolling average of historical errors
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df['error'].rolling(window=window).mean()
                df[f'abs_error_rolling_{window}d'] = df['abs_error'].rolling(window=window).mean()
            else:
                # Otherwise fill with 0
                df[f'error_rolling_{window}d'] = 0
                df[f'abs_error_rolling_{window}d'] = 0
        
        # Calculate financial ratios using the correct column names
        revenue_col = 'INCOME_STATEMENT_totalRevenue'
        profit_col = 'INCOME_STATEMENT_grossProfit'
        cost_col = 'INCOME_STATEMENT_costOfRevenue'
        
        if revenue_col in df.columns and profit_col in df.columns:
            try:
                # Calculate year-over-year growth rates for quarterly revenue and profit
                df['revenue_yoy_growth'] = df.groupby(['quarter'])[revenue_col].pct_change(4, fill_method=None)
                df['profit_yoy_growth'] = df.groupby(['quarter'])[profit_col].pct_change(4, fill_method=None)
                
                # Remove outliers
                df['revenue_yoy_growth'] = df['revenue_yoy_growth'].apply(
                    lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                )
                df['profit_yoy_growth'] = df['profit_yoy_growth'].apply(
                    lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                )
                
                # Calculate profit margin
                df['profit_margin'] = df.apply(
                    lambda row: row[profit_col] / row[revenue_col] if row[revenue_col] != 0 else 0, 
                    axis=1
                )
                
                if cost_col in df.columns:
                    # Try to extract relative magnitude of financial data
                    df['revenue_to_cost_ratio'] = df.apply(
                        lambda row: row[revenue_col] / row[cost_col] if row[cost_col] != 0 else 1, 
                        axis=1
                    )
            except Exception as e:
                print(f"Error calculating financial ratios: {e}")
                # If calculation fails, use default values
                df['revenue_yoy_growth'] = 0
                df['profit_yoy_growth'] = 0
                df['profit_margin'] = 0
                df['revenue_to_cost_ratio'] = 1
        
        # Add macroeconomic indicators (if they exist)
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL']
        for col in macro_cols:
            if col in df.columns:
                try:
                    # Create some simple derived features
                    df[f'{col}_lag1'] = df[col].shift(1)
                    df[f'{col}_change'] = df[col].pct_change(fill_method=None)
                    
                    # Replace NaN and infinity with 0
                    df[f'{col}_change'] = df[f'{col}_change'].apply(
                        lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                    )
                except:
                    df[f'{col}_lag1'] = df[col]
                    df[f'{col}_change'] = 0
        
        # Add lagged features - using available recent historical data
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df['prediction'].shift(lag)
            
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df['error'].shift(lag)
            else:
                df[f'error_lag_{lag}'] = 0
        
        # For the first date of future predictions, lagged features may need to be obtained from historical data
        # Simplified handling here, filling missing values
        # Fill rolling and lagged features
        for window in [3, 7, 14]:
            df[f'prediction_rolling_{window}d'] = df[f'prediction_rolling_{window}d'].fillna(df['prediction'])
            df[f'error_rolling_{window}d'] = df[f'error_rolling_{window}d'].fillna(0)
            df[f'abs_error_rolling_{window}d'] = df[f'abs_error_rolling_{window}d'].fillna(0)
            
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df[f'prediction_lag_{lag}'].fillna(df['prediction'])
            df[f'error_lag_{lag}'] = df[f'error_lag_{lag}'].fillna(0)
            
        # Use forward fill for other financial and macro data
        for col in df.columns:
            if col not in ['Date', 'prediction'] and df[col].isnull().any():
                df[col] = df[col].ffill().bfill().fillna(0)
                
        # Final check: replace any NaN or infinity values
        for col in df.columns:
            if col != 'Date':  # Don't process date column
                try:
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
                    df[col] = df[col].fillna(0)
                except:
                    pass
                
        return df
    
    def predict_next_quarters(self, num_quarters=4):
        """
        Predict financial data for future quarters
        Simple method: using time series prediction or average growth rate
        """
        # Get recent quarterly data
        recent_quarters = self.quarterly_df.sort_values('fiscalDateEnding').tail(8)
        
        # Calculate average quarterly growth rates
        growth_rates = {}
        for col in recent_quarters.columns:
            if col in ['Ticker', 'fiscalDateEnding', 'observation_date'] or 'reportedCurrency' in col:
                continue
                
            # Only calculate growth rates for numeric columns
            if recent_quarters[col].dtype in [np.float64, np.int64]:
                try:
                    # Calculate quarter-over-quarter growth rates
                    pct_changes = recent_quarters[col].pct_change().dropna()
                    # Take the average growth rate
                    if not pct_changes.empty:
                        growth_rates[col] = pct_changes.mean()
                except:
                    # If calculation fails, set to 0
                    growth_rates[col] = 0
        
        # Get the latest quarterly data
        last_quarter = recent_quarters.iloc[-1].copy()
        next_quarters = []
        
        # Predict future quarters
        for i in range(1, num_quarters+1):
            # Copy the previous quarter's data
            next_quarter = last_quarter.copy()
            
            # Calculate date for next quarter
            next_date = pd.to_datetime(last_quarter['fiscalDateEnding']) + pd.DateOffset(months=3*i)
            next_quarter['fiscalDateEnding'] = next_date
            
            # Update financial metrics, applying growth rates
            for col, rate in growth_rates.items():
                if pd.notna(last_quarter[col]) and last_quarter[col] != 0:
                    next_quarter[col] = last_quarter[col] * (1 + rate)
            
            # Add to predicted quarters list
            next_quarters.append(next_quarter)
        
        # Convert to DataFrame
        next_quarters_df = pd.DataFrame(next_quarters)
        return next_quarters_df

    
    def run_pipeline(self, daily_predictions_path, quarterly_data_path, future_days=None, test_size=0.2):
        """
        Run the complete fusion model pipeline
        
        Parameters:
        daily_predictions_path: Path to daily prediction data
        quarterly_data_path: Path to quarterly data
        future_days: If provided, will predict for this many days in the future
        test_size: Train-test split ratio
        """
        # Get the logger
        logger = logging.getLogger()
        
        logger.info("1. Loading data...")
        self.load_data(daily_predictions_path, quarterly_data_path)
        
        logger.info("\n2. Merging quarterly data into daily data...")
        self.merge_quarterly_to_daily()
        
        # Check columns in the merged data
        logger.info("\nData column preview:")
        for col in sorted(self.merged_df.columns):
            if 'Date' in col or 'prediction' in col or 'actual' in col or 'error' in col:
                logger.info(f"- {col}")
                
        # Check if necessary columns exist
        required_cols = ['Date', 'prediction']
        for col in required_cols:
            if col not in self.merged_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        logger.info("\n3. Creating fusion features...")
        self.create_features()
        
        # If row count is 0 after feature creation, try basic features
        if len(self.feature_df) == 0:
            logger.warning("Warning: No data after feature engineering, trying to use basic features...")
            df = self.merged_df.copy()
            
            # Keep only basic features
            basic_cols = ['Date', 'prediction']
            
            if 'actual' in df.columns:
                basic_cols.append('actual')
                
            # Add time features
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
            
            # Use basic features as the feature set
            self.feature_df = df[basic_cols + ['day_of_week', 'month', 'quarter']]
            logger.info(f"Row count after using basic features: {len(self.feature_df)}")
        
        logger.info("\n4. Training fusion model...")
        try:
            evaluation = self.train_model(test_size=test_size)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Use simple linear regression as a fallback
            logger.info("Trying simple linear regression as a fallback model...")
            from sklearn.linear_model import LinearRegression
            
            # Use the most basic features
            basic_df = self.merged_df[['Date', 'prediction']].copy()
            basic_df['day_of_week'] = basic_df['Date'].dt.dayofweek
            basic_df['month'] = basic_df['Date'].dt.month
            
            if 'actual' in self.merged_df.columns:
                basic_df['actual'] = self.merged_df['actual']
                
            self.feature_df = basic_df.dropna()
            
            # Use prediction as the only feature
            X = self.feature_df[['prediction', 'day_of_week', 'month']]
            y = self.feature_df['actual']
            
            # Train simple model
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.model_type = 'linear'
            
            # Simple evaluation
            preds = self.model.predict(X)
            mse = mean_squared_error(y, preds)
            logger.info(f"Simple linear model MSE: {mse:.6f}")
        
        logger.info("\n5. Generating final predictions for historical data...")
        final_predictions = self.predict()
        
        # Add prediction results to original data
        result_df = self.feature_df.copy()
        result_df['fusion_prediction'] = final_predictions
        
        if 'actual' in result_df.columns:
            result_df['fusion_error'] = result_df['fusion_prediction'] - result_df['actual']
            result_df['fusion_abs_error'] = np.abs(result_df['fusion_error'])
            
            if 'prediction' in result_df.columns:
                result_df['deepar_error'] = result_df['prediction'] - result_df['actual']
                result_df['deepar_abs_error'] = np.abs(result_df['deepar_error'])
                
                # Calculate improvement of fusion model over DeepAR
                avg_deepar_error = result_df['deepar_abs_error'].mean()
                avg_fusion_error = result_df['fusion_abs_error'].mean()
                improvement = (avg_deepar_error - avg_fusion_error) / avg_deepar_error * 100
                
                logger.info(f"\nFusion model average absolute error: {avg_fusion_error:.6f}")
                logger.info(f"DeepAR model average absolute error: {avg_deepar_error:.6f}")
                logger.info(f"Relative improvement: {improvement:.2f}%")
        
        # Keep only Date and fusion_prediction columns
        output_df = result_df[['Date', 'fusion_prediction', 'actual']].copy()
        
        # If future prediction is needed
        if future_days is not None:
            logger.info("\n6. Generating future predictions...")
            # Create future dates
            last_date = self.daily_df['Date'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
            
            # Create future dataframe
            # This assumes that the DeepAR model has already generated predictions for these dates
            # If not, use the last prediction value as a placeholder
            last_prediction = self.daily_df['prediction'].iloc[-1]
            future_df = pd.DataFrame({
                'Date': future_dates,
                'prediction': [last_prediction] * future_days  # Placeholder prediction value
            })
            
            # Predict future
            future_predictions = self.predict_future(future_df)
            future_output = future_predictions[['Date', 'fusion_prediction', 'actual']].copy()
            
            # Merge historical and future predictions
            all_results = pd.concat([output_df, future_output])
            return all_results
        
        return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fusion Layer for Time Series Prediction')
    parser.add_argument('--daily', type=str, required=True, help='Path to daily predictions CSV')
    parser.add_argument('--quarterly', type=str, required=True, help='Path to quarterly data CSV')
    parser.add_argument('--future', type=int, default=0, help='Number of future days to predict')
    parser.add_argument('--output', type=str, default='fusion_predictions.csv', help='Output file path')
    parser.add_argument('--lookback', type=int, default=7, help='Number of lookback days for features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test data')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.daily)
    log_file = os.path.join(output_dir, 'fusion_predictions.log')
    setup_logging(log_file)
    
    fusion = FusionLayer(lookback_days=args.lookback)
    
    results = fusion.run_pipeline(
        daily_predictions_path=args.daily,
        quarterly_data_path=args.quarterly,
        future_days=args.future if args.future > 0 else None,
        test_size=args.test_size
    )
    
    output_file = os.path.join(output_dir, args.output)
    
    results.to_csv(output_file, index=False)
    logging.info(f"\nResults have been saved to {output_file}")