import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from pandas_datareader import data as pdr
import os
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
import json

# Define directories
BASE_DIR = "data"
STOCK_DIR = os.path.join(BASE_DIR, "stock")
FINANCIAL_DIR = os.path.join(BASE_DIR, "financial")
FRED_DIR = os.path.join(BASE_DIR, "economic")
CDNOD_DIR = os.path.join("cdnod+nowcasting", "cdnod_graph")

# Create directories if they don't exist
os.makedirs(STOCK_DIR, exist_ok=True)
os.makedirs(FINANCIAL_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)
os.makedirs(CDNOD_DIR, exist_ok=True)

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)
    if not data.empty:
        data['Ticker'] = ticker
        data.reset_index(inplace=True)
        data["Return"] = data["Close"].pct_change()
        data = data.apply(pd.to_numeric, errors='coerce')
    return data

# Fetch economic indicators
def fetch_fred_data(series, start_date):
    fred_data = pdr.DataReader(series, "fred", start=start_date)
    fred_data.reset_index(inplace=True)
    fred_data["DATE"] = pd.to_datetime(fred_data["DATE"])
    return fred_data.ffill()

# Fetch company financials
def fetch_company_financials(ticker, api_key):
    report_types = {"INCOME_STATEMENT": "MonthlyReports", "BALANCE_SHEET": "MonthlyReports", "CASH_FLOW": "MonthlyReports"}
    report_data = {}
    for report_type, key in report_types.items():
        url = f"https://www.alphavantage.co/query?function={report_type}&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if key in data:
            df = pd.DataFrame(data[key])
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
            df = df[df["fiscalDateEnding"] >= "2019-01-01"]
            for _, row in df.iterrows():
                date_key = row["fiscalDateEnding"]
                if date_key not in report_data:
                    report_data[date_key] = {"Ticker": ticker, "fiscalDateEnding": date_key}
                for col in row.index:
                    if col not in ["fiscalDateEnding"]:
                        report_data[date_key][f"{report_type}_{col}"] = row[col]
    return pd.DataFrame(report_data.values())

if __name__ == "__main__":
    tickers = ["AMZN","GOOG","ABT","CVS","AMGN","T"]
    start_date = "2019-09-30"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    for ticker in tickers:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        stock_data.to_csv(os.path.join(STOCK_DIR, f"{ticker.lower()}_stock_data.csv"), index=False)
    
    fred_series = ["M2SL", "M1SL", "FEDFUNDS", "PPIACO", "RTWEXBGS", "CPIAUCSL", "UNRATE", "GDP"]
    fred_data = fetch_fred_data(fred_series, "2019-01-01")
    fred_data.to_csv(os.path.join(FRED_DIR, "fred_data.csv"), index=False)
    
    # api_key = "C0E1JBHIL5VPHPX4"
    # for ticker in tickers:
    #     financial_data = fetch_company_financials(ticker, api_key)
    #     financial_data.to_csv(os.path.join(FINANCIAL_DIR, f"{ticker.lower()}_financial_data.csv"), index=False)
    
    dict = {}
    for ticker in tickers:
        if ticker == "GOOG" or ticker == "AMZN":
            corr = 0.95
        elif ticker == "T":
            corr = 0.60
        else:
            corr = 0.75
        financial_data = pd.read_csv(os.path.join(FINANCIAL_DIR, f"{ticker.lower()}_financial_data.csv"))
        financial_data["fiscalDateEnding"] = pd.to_datetime(financial_data["fiscalDateEnding"])
        financial_data.replace("None", np.nan, inplace=True)
        financial_data = financial_data.dropna(axis=1, how='any')
        financial_data = financial_data.drop(columns = ["BALANCE_SHEET_shortTermDebt","CASH_FLOW_reportedCurrency","BALANCE_SHEET_reportedCurrency","INCOME_STATEMENT_reportedCurrency","Ticker"])
        for col in financial_data.columns[2:]:
          financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce').astype('Int64') 
        group = []
        financial_data["fiscalDateEnding"] = pd.to_datetime(financial_data["fiscalDateEnding"])
        for i in financial_data["fiscalDateEnding"]:
            year = str(i.year)      
            if (i.month == 12):
                group.append(year + "-02-01")
            elif (i.month == 3):
                group.append(year + "-05-01")
            elif (i.month == 6):
                group.append(year + "-08-01")
            elif (i.month == 9):
                group.append(year + "-11-01")
        financial_data.drop(columns = ["fiscalDateEnding"], inplace=True)
        financial_data["group"] = pd.to_datetime(group)
        
        stock = pd.read_csv(os.path.join(STOCK_DIR, f"{ticker.lower()}_stock_data.csv"))
        stock["Date"] = pd.to_datetime(stock["Date"]).dt.date 
        stock["Date"] = pd.to_datetime(stock["Date"])  

        target = stock[["Date","Close"]]
        target = stock[["Date", "Close"]].iloc[2:].reset_index(drop=True)
        merged_data = pd.merge_asof(
            target.sort_values("Date"),
            financial_data.sort_values("group"),
            left_on="Date",
            right_on="group",
            direction="backward"
        )

        merged_data = pd.merge_asof(
            merged_data.sort_values("Date"),
            fred_data.sort_values("DATE"),
            left_on="Date",
            right_on="DATE",
            direction="backward"
        )

        selected_columns = [i for i in merged_data.columns if "_".join(i.split("_")[:2]) in {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}]
        merged_data[selected_columns] = merged_data[selected_columns] / 1e9  

        # Convert 'Date' to Months and create integer labels
        merged_data["Date"] = pd.to_datetime(merged_data["Date"])
        merged_data["Month"] = merged_data["Date"].dt.to_period("M")
        merged_data["Month_Label"] = pd.factorize(merged_data["Month"])[0] + 1

        # Define grouping variable
        group_column = "Month_Label"

        # Drop non-numeric columns
        amgn_preprocessed = merged_data.drop(columns=["Date", "Month", "Month_Label","DATE","group"])
        column_mapping = {f"x{i+1}": col for i, col in enumerate(amgn_preprocessed.columns)}
        dict[ticker] = column_mapping
        
        
        # Remove highly correlated columns
        correlation_matrix = amgn_preprocessed.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr)]
        amgn_preprocessed = amgn_preprocessed.drop(columns=to_drop)
        amgn_preprocessed["Close"] = amgn_preprocessed["Close"].astype(float)

        # Convert data to NumPy array
        X = amgn_preprocessed.to_numpy()
        c_indx = [[i] for i in merged_data["Month_Label"]]
        X = np.array(X, dtype=np.float64)
        
        # Validate dimensions
        print(f"Shape of X: {X.shape}, Max index in c_indx: {max(max(c_indx))}")

        # Run CD-NOD causal discovery
        try:
            cg = cdnod(X, c_indx=c_indx, indep_test=fisherz, alpha=0.01)

            # Save causal graph
            pyd = GraphUtils.to_pydot(cg.G)
            graph_image_path = os.path.join(os.path.join(CDNOD_DIR, f'{ticker.lower()}_fisherz_M.png'))
            pyd.write_png(graph_image_path)
            print(f"Graph saved at: {graph_image_path}")


        except Exception as e:
            print(f"CD-NOD failed: {e}")
    
    with open(os.path.join(CDNOD_DIR, "cdnod_label.jsonl"), "w") as f:
        json.dump(dict, f, indent=4)
                        
                        
                        

                