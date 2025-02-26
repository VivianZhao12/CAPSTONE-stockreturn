import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = "data"
STOCK_DIR = os.path.join("..", BASE_DIR, "stock")
FINANCIAL_DIR = os.path.join("..", BASE_DIR, "financial")
FRED_DIR = os.path.join("..", BASE_DIR, "economic")
CDNOD_DIR = os.path.join("..", "..","macro+micro-regression","cdnod", "cdnod_graph")
aligned_macromicro_DIR = os.path.join("..", BASE_DIR, "macro_micro")


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

    # Select independent variables for standardization
    independent_vars = df[['M2SL', 'M1SL', 'FEDFUNDS', 'PPIACO', 'RTWEXBGS', 'CPIAUCSL', 'UNRATE', 'GDP']]
    
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the data
    standardized_vars = scaler.fit_transform(independent_vars)
    
    # Create a DataFrame with standardized variables
    standardized_df = pd.DataFrame(standardized_vars, columns=independent_vars.columns)
    standardized_df['DATE'] = df['DATE']
    
    # Convert date column to datetime
    standardized_df["DATE"] = pd.to_datetime(standardized_df["DATE"])
    
    # Extract year and month
    standardized_df["year"] = standardized_df["DATE"].dt.year
    standardized_df["month"] = standardized_df["DATE"].dt.month
    
    # Identify quarters
    standardized_df["quarter"] = standardized_df["DATE"].dt.to_period("Q")

    # Determine the position of the month in the quarter
    standardized_df["month_position"] = standardized_df["month"] % 3  # 0 = first month, 1 = second month, 2 = third month

    # Create separate DataFrames for each month's position in the quarter
    df_first_month = standardized_df[standardized_df["month_position"] == 1].copy()
    df_second_month = standardized_df[standardized_df["month_position"] == 2].copy()
    df_third_month = standardized_df[standardized_df["month_position"] == 0].copy()  

    # Merge by quarter
    df_quarterly = df_first_month[["quarter", "M2SL", "M1SL", "FEDFUNDS", "PPIACO",
                                   "RTWEXBGS", "UNRATE", "CPIAUCSL"]].rename(
        columns=lambda x: x + "_M1" if x != "quarter" else x)

    df_quarterly = df_quarterly.merge(
        df_second_month[["quarter", "M2SL", "M1SL", "FEDFUNDS", "PPIACO",
                         "RTWEXBGS", "UNRATE", "CPIAUCSL"]].rename(
            columns=lambda x: x + "_M2" if x != "quarter" else x),
        on="quarter", how="inner"
    )

    df_quarterly = df_quarterly.merge(
        df_third_month[["quarter", "M2SL", "M1SL", "FEDFUNDS", "PPIACO",
                        "RTWEXBGS", "UNRATE", "CPIAUCSL", "GDP"]].rename(
            columns=lambda x: x + "_M3" if x not in ["quarter", "GDP"] else x),
        on="quarter", how="inner"
    )

    return df_quarterly


def hypothesis_test(merged_data):
    dependent_var = "Quarterly_Return"
    independent_vars = ['M2SL', 'M1SL', 'FEDFUNDS', 'PPIACO', 'RTWEXBGS', 'CPIAUCSL', 'UNRATE']
    best_month_results = {}

    for var in independent_vars:
        months = ["M1", "M2", "M3"]
        results = {}

        for month in months:
            X = merged_data[[f"{var}_{month}"]]
            X = sm.add_constant(X)
            y = merged_data[dependent_var]
            model = sm.OLS(y, X).fit()
            results[month] = {
                "p_value": model.pvalues.get(f"{var}_{month}", float("inf")),
                "R2": model.rsquared
            }
        best_month = min(results, key=lambda m: results[m]["p_value"])
        best_result = results[best_month]
        best_month_results[var] = {
            "Best Month": best_month,
            "P-Value": best_result["p_value"],
            "R2": best_result["R2"]
        }
    best_month_df = pd.DataFrame.from_dict(best_month_results, orient="index")
    selected_months = best_month_df["Best Month"].to_dict()
    selected_columns = ["quarter", "GDP"]

    for var, best_month in selected_months.items():
        selected_columns.append(f"{var}_{best_month}")

    final_dataset = merged_data[selected_columns].copy()
    final_dataset.columns = [col.replace("_M1", "").replace("_M2", "").replace("_M3", "") for col in final_dataset.columns]
    final_dataset["observation_date"] = final_dataset["quarter"].dt.end_time.dt.strftime("%Y-%m-%d")
    ordered_columns = ["observation_date"] + [col for col in final_dataset.columns if col not in ["quarter", "observation_date"]]
    final_dataset = final_dataset[ordered_columns]
    return final_dataset



if __name__ == '__main__':
    
    # Configuration:
    for ticker in ["AMZN","GOOG","ABT","CVS","AMGN","T"]:
        
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



