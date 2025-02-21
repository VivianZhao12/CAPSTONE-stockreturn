import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


def fetch_stock_return(ticker):
    all_data = pd.DataFrame(columns=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])

    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    data['Ticker'] = ticker  # Add the ticker column
    data.reset_index(inplace=True)  # Reset the index to include the date column
    data.columns = ["Date","Adj Close", "Close", "High", "Low", "Open", "Volume", 'Ticker']
    all_data = pd.concat([data, all_data], ignore_index=True)

    testing_data = pd.DataFrame()
    testing_data['Daily_Return'] = (all_data['Close'] - all_data['Close'].shift(1)) / all_data['Close'].shift(1)
    testing_data['Date'] = all_data['Date']
    # Calculate Quarterly Returns
    testing_data["Date"] = pd.to_datetime(testing_data["Date"])
    testing_data["quarter"] = testing_data["Date"].dt.to_period("Q")

    quarterly_returns = (
        testing_data.groupby("quarter")["Daily_Return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
        .rename(columns={"Daily_Return": "Quarterly_Return"})
    )
    return quarterly_returns

def hypothesis_prep(df):
    independent_vars = df[['Money_Supply_M2', 'Money_Supply_M1',
        'Interest_Rate', 'PPI', 'Real_Dollar_Index', 'Unemployment_Rate', 'CPI',
        'GDP']]
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the data
    standardized_vars = scaler.fit_transform(independent_vars)
    # Create a DataFrame with standardized variables
    standardized_df = pd.DataFrame(standardized_vars, columns=independent_vars.columns)
    standardized_df['observation_date'] = df['observation_date']
    df = standardized_df
    # Convert date column to datetime
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    # Extract year and month
    df["year"] = df["observation_date"].dt.year
    df["month"] = df["observation_date"].dt.month
    # Identify quarters
    df["quarter"] = df["observation_date"].dt.to_period("Q")

    df["month_position"] = df["month"] % 3  # 0 = first month, 1 = second month, 2 = third month

    # Create separate DataFrames for each month's position in the quarter
    df_first_month = df[df["month_position"] == 1].copy()
    df_second_month = df[df["month_position"] == 2].copy()
    df_third_month = df[df["month_position"] == 0].copy()  

    # Merge by quarter 
    df_quarterly = df_first_month[["quarter", "Money_Supply_M2", "Money_Supply_M1", "Interest_Rate", "PPI",
                                "Real_Dollar_Index", "Unemployment_Rate", "CPI"]].rename(
        columns=lambda x: x + "_M1" if x != "quarter" else x)
    df_quarterly = df_quarterly.merge(
        df_second_month[["quarter", "Money_Supply_M2", "Money_Supply_M1", "Interest_Rate", "PPI",
                        "Real_Dollar_Index", "Unemployment_Rate", "CPI"]].rename(
            columns=lambda x: x + "_M2" if x != "quarter" else x),
        on="quarter", how="inner"
    )
    df_quarterly = df_quarterly.merge(
        df_third_month[["quarter", "Money_Supply_M2", "Money_Supply_M1", "Interest_Rate", "PPI",
                        "Real_Dollar_Index", "Unemployment_Rate", "CPI", "GDP"]].rename(
            columns=lambda x: x + "_M3" if x not in ["quarter", "GDP"] else x),
        on="quarter", how="inner"
    )
    return df_quarterly

def hypothesis_test(merged_data):
    # Define the dependent variable and exclude non-numeric columns
    dependent_var = "Quarterly_Return"
    exclude_columns = ["Quarter", "observation_date", dependent_var]  # Add other non-numeric columns here

    # Filter numeric independent variables
    independent_vars = ['Money_Supply_M2', 'Money_Supply_M1', 'Interest_Rate', 'PPI',
        'Real_Dollar_Index', 'CPI', 'Unemployment_Rate']


    # Initialize a dictionary to store results
    best_month_results = {}

    # Loop through each independent variable
    for var in independent_vars:
        months = ["M1", "M2", "M3"]
        results = {}

        # Loop through each month
        for month in months:
            # Select the variable for the specific month
            X = merged_data[[f"{var}_{month}"]]
            X = sm.add_constant(X)
            y = merged_data[dependent_var]
            
            # Fit the regression model
            model = sm.OLS(y, X).fit()
            
            # Store p-value and R^2
            results[month] = {
                "p_value": model.pvalues.get(f"{var}_{month}", float("inf")),
                "R2": model.rsquared,
                "Summary": model.summary()
            }
        
        # Identify the best month (based on lowest p-value or highest R^2)
        best_month = min(results, key=lambda m: results[m]["p_value"])
        best_result = results[best_month]
        
        # Store the best month's result for this variable
        best_month_results[var] = {
            "Best Month": best_month,
            "P-Value": best_result["p_value"],
            "R2": best_result["R2"]
        }

    # Convert the results to a DataFrame for better readability
    best_month_df = pd.DataFrame.from_dict(best_month_results, orient="index")

    # Extract the best month for each variable from the hypothesis test results
    selected_months = best_month_df["Best Month"].to_dict()  # Dictionary mapping variable to its best month

    # Create the final dataset with only the selected month's data
    selected_columns = ["quarter", "GDP"]  # Start with essential columns

    for var, best_month in selected_months.items():
        selected_columns.append(f"{var}_{best_month}")  # Add the best month's column for the variable

    # Create final dataset using selected columns
    final_dataset = merged_data[selected_columns].copy()

    # Rename columns to remove the "_M1", "_M2", "_M3" suffix for clarity
    final_dataset.columns = [col.replace("_M1", "").replace("_M2", "").replace("_M3", "") for col in final_dataset.columns]

    # Convert quarter to end-of-quarter date format
    final_dataset["observation_date"] = final_dataset["quarter"].dt.end_time.dt.strftime("%Y-%m-%d")

    # Reorder columns for better readability
    ordered_columns = ["observation_date"] + [col for col in final_dataset.columns if col not in ["quarter", "observation_date"]]
    final_dataset = final_dataset[ordered_columns]
    return final_dataset



if __name__ == '__main__':
    # manually update macro_monthly_data from the following sources:
    # M2: https://fred.stlouisfed.org/series/M2SL
    # M1: https://fred.stlouisfed.org/series/M1SL
    # IR: https://fred.stlouisfed.org/series/FEDFUNDS
    # PPI: https://fred.stlouisfed.org/series/PPIACO
    # DI: https://fred.stlouisfed.org/series/RTWEXBGS
    # CPI: https://fred.stlouisfed.org/series/CPIAUCSL
    # UN: https://fred.stlouisfed.org/series/UNRATE
    # GDP: https://fred.stlouisfed.org/series/GDP

    # Configuration:
    ticker = ['ABT'] # company to do hypothesis testing on
    data_path = 'CAPSTONE-stockreturn/Data/macro_micro/macro_monthly_data.csv' # data path for original monthly data
    output_path = f"{ticker[0]}_macro_quarterly"
    start_date = "2019-12-01"
    end_date = datetime.today().strftime('2024-7-31') # set this to today's date if you have full data up to date

    # prepare macro data
    df = pd.read_csv(data_path) 

    quarterly_returns = fetch_stock_return(ticker)
    df_quarterly = hypothesis_prep(df)
    merged_data = pd.merge(quarterly_returns, df_quarterly, on="quarter", how="inner")

    final_df = hypothesis_test(merged_data)

    final_df.to_csv(output_path, index=False)



