
import pandas as pd

def create_df(features, ticker):
    """"take in a list of features and creates a dataframe with those features for a specific company"""
    data_path = f'../data/macro_micro/{ticker}_quarterly.csv'
    macro_features = ['M2SL', 'M1SL', 'FEDFUNDS', 'PPIACO', 'RTWEXBGS', 'CPIAUCSL', 'UNRATE']
    all_selected_features = macro_features + features + ['Quarterly_Return'] + ['fiscalDateEnding']
    df = pd.read_csv(data_path)
    return df[all_selected_features]

if __name__ == "__main__":
    # define tickers and either selected variables from cdnod.py
    features = {'amzn':["M1SL","M2SL","INCOME_STATEMENT_operatingIncome","CASH_FLOW_cashflowFromInvestment","CASH_FLOW_capitalExpenditures","CASH_FLOW_operatingCashflow","CASH_FLOW_changeInOperatingLiabilities","BALANCE_SHEET_cashAndCashEquivalentsAtCarryingValue"],
                'amgn':['CASH_FLOW_changeInOperatingAssets','BALANCE_SHEET_otherCurrentAssets', 'UNRATE','CASH_FLOW_operatingCashflow', 'BALANCE_SHEET_currentDebt'],
                'goog':['INCOME_STATEMENT_incomeTaxExpense','INCOME_STATEMENT_otherNonOperatingIncome','M2SL','CPIAUCSL','CASH_FLOW_paymentsForRepurchaseOfCommonStock','UNRATE','PPIACO','M1SL','CASH_FLOW_cashflowFromFinancing'],
                'abt':['INCOME_STATEMENT_interestAndDebtExpense','CASH_FLOW_proceedsFromRepaymentsOfShortTermDebt','CASH_FLOW_changeInOperatingAssets','CASH_FLOW_depreciationDepletionAndAmortization','CASH_FLOW_changeInReceivables','BALANCE_SHEET_currentDebt'],
                'cvs':['BALANCE_SHEET_totalNonCurrentLiabilities','BALANCE_SHEET_inventory','BALANCE_SHEET_currentDebt','UNRATE','INCOME_STATEMENT_investmentIncomeNet','CASH_FLOW_paymentsForOperatingActivities','BALANCE_SHEET_otherCurrentLiabilities','BALANCE_SHEET_shortTermInvestments', 'CASH_FLOW_depreciationDepletionAndAmortization','INCOME_STATEMENT_nonInterestIncome','INCOME_STATEMENT_interestAndDebtExpense','CASH_FLOW_capitalExpenditures','CASH_FLOW_proceedsFromRepaymentsOfShortTermDebt','BALANCE_SHEET_cashAndCashEquivalentsAtCarryingValue','CASH_FLOW_cashflowFromFinancing','BALANCE_SHEET_propertyPlantEquipment','CASH_FLOW_changeInOperatingAssets','BALANCE_SHEET_otherNonCurrentAssets','CASH_FLOW_changeInOperatingLiabilities','CASH_FLOW_changeInInventory','CASH_FLOW_proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet','BALANCE_SHEET_intangibleAssetsExcludingGoodwill','CASH_FLOW_paymentsForRepurchaseOfCommonStock', 'INCOME_STATEMENT_totalRevenue','CASH_FLOW_cashflowFromInvestment'],
                't':['BALANCE_SHEET_treasuryStock','BALANCE_SHEET_propertyPlantEquipment', 'BALANCE_SHEET_capitalLeaseObligations','BALANCE_SHEET_inventory','CASH_FLOW_paymentsForRepurchaseOfCommonStock','INCOME_STATEMENT_operatingIncome', 'INCOME_STATEMENT_totalRevenue']
    }
    for ticker in features.keys():
        save_path = f'../data/macro_micro/{ticker}_quarterly_cdnod.csv'
        output = create_df(features[ticker], ticker)
        output.to_csv(save_path, index=False)