import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os



def conver_to_quarterly_return(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data["quarter"] = data["Date"].dt.to_period("Q")

    quarterly_returns = (
        data.groupby("quarter")["Daily Return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
        .rename(columns={"Daily Return": "Quarterly_Return"})
    )
    return quarterly_returns

def hypothesis_prep(df):
    # Convert DATE column to datetime
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Rename columns to more meaningful names
    df = df.rename(columns={
        "DATE": "observation_date",
        "M2SL": "Money_Supply_2",
        "M1SL": "Money_Supply_1",
        "FEDFUNDS": "Interest_Rate",
        "PPIACO": "PPI",
        "RTWEXBGS": "Real_Dollar_Index",
        "CPIAUCSL": "CPI",
        "UNRATE": "Unemployment_Rate",
        "GDP": "GDP"
    })

    # Select independent variables for standardization
    independent_vars = df[['Money_Supply_2', 'Money_Supply_1',
        'Interest_Rate', 'PPI', 'Real_Dollar_Index', 'Unemployment_Rate', 'CPI',
        'GDP']]
    
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the data
    standardized_vars = scaler.fit_transform(independent_vars)
    
    # Create a DataFrame with standardized variables
    standardized_df = pd.DataFrame(standardized_vars, columns=independent_vars.columns)
    standardized_df['observation_date'] = df['observation_date']
    
    # Convert date column to datetime
    standardized_df["observation_date"] = pd.to_datetime(standardized_df["observation_date"])
    
    # Extract year and month
    standardized_df["year"] = standardized_df["observation_date"].dt.year
    standardized_df["month"] = standardized_df["observation_date"].dt.month
    
    # Identify quarters
    standardized_df["quarter"] = standardized_df["observation_date"].dt.to_period("Q")

    # Determine the position of the month in the quarter
    standardized_df["month_position"] = standardized_df["month"] % 3  # 0 = first month, 1 = second month, 2 = third month

    # Create separate DataFrames for each month's position in the quarter
    df_first_month = standardized_df[standardized_df["month_position"] == 1].copy()
    df_second_month = standardized_df[standardized_df["month_position"] == 2].copy()
    df_third_month = standardized_df[standardized_df["month_position"] == 0].copy()  

    # Merge by quarter
    df_quarterly = df_first_month[["quarter", "Money_Supply_2", "Money_Supply_1", "Interest_Rate", "PPI",
                                   "Real_Dollar_Index", "Unemployment_Rate", "CPI"]].rename(
        columns=lambda x: x + "_M1" if x != "quarter" else x)

    df_quarterly = df_quarterly.merge(
        df_second_month[["quarter", "Money_Supply_2", "Money_Supply_1", "Interest_Rate", "PPI",
                         "Real_Dollar_Index", "Unemployment_Rate", "CPI"]].rename(
            columns=lambda x: x + "_M2" if x != "quarter" else x),
        on="quarter", how="inner"
    )

    df_quarterly = df_quarterly.merge(
        df_third_month[["quarter", "Money_Supply_2", "Money_Supply_1", "Interest_Rate", "PPI",
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
    independent_vars = ['Money_Supply_2', 'Money_Supply_1', 'Interest_Rate', 'PPI',
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
    selected_columns = ["quarter", "GDP",'Quarterly_Return']  # Start with essential columns

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
    # Configuration:
    BASE_DIR = "../data"
    STOCK_DIR = os.path.join(BASE_DIR, "stock")
    FINANCIAL_DIR = os.path.join(BASE_DIR, "financial")
    FRED_DIR = os.path.join(BASE_DIR, "economic")
    CDNOD_DIR = os.path.join("macro+micro_regressio/cdnod", "cdnod_graph")
    aligned_macromicro_DIR = os.path.join(BASE_DIR, "macro_micro")
    tickers = ["AMZN","GOOG","ABT","CVS","AMGN","T"]

    for ticker in tickers:
        
        data_path = os.path.join(FRED_DIR, "fred_data.csv")
        stock = pd.read_csv(os.path.join(STOCK_DIR, f'{ticker.lower()}_stock_data.csv'))
        output_path = os.path.join(aligned_macromicro_DIR, f"{ticker.lower()}_quarterly.csv")
        start_date = "2018-12-01"
        end_date = datetime.today().strftime('%Y-%m-%d')

        # prepare macro data
        df = pd.read_csv(data_path)
        quarterly_returns = conver_to_quarterly_return(stock)
        df_quarterly = hypothesis_prep(df)
        merged_data = pd.merge(quarterly_returns, df_quarterly, on="quarter", how="inner")

        final_df = hypothesis_test(merged_data)
        
        company_data = pd.read_csv(os.path.join(FINANCIAL_DIR, f'{ticker.lower()}_financial_data.csv'))
        company_data["fiscalDateEnding"] = pd.to_datetime(company_data["fiscalDateEnding"])
        final_df["observation_date"] = pd.to_datetime(final_df["observation_date"])
        
        final_df = company_data.merge(final_df, left_on="fiscalDateEnding", right_on="observation_date")

        final_df.to_csv(output_path, index=False)



