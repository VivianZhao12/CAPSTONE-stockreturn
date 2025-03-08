# Causal Discovery in Stock Return

This repository contains the implementation of a multi-model framework for stock return prediction that integrates sentiment analysis, historical stock data, and macroeconomic indicators. The project uses DeepAR for time series prediction, FinBERT for sentiment analysis, and Random Forest Regressor for economic indicators.


## Project Overview
Our framework combines three main components:
- Time series prediction using DeepAR and PCMCI
- Sentiment analysis using FinBERT
- Economic impact analysis using CDNOD and Random Forest Regressor

## Project Structure
```bash
CAPSTONE-stockreturn/
├── DeepAR/                    # DeepAR model implementation
│   ├── experiments/           # Experiment results for each company
│   └── README.md              # DeepAR specific documentation
├── FinBERT/                   # FinBERT implementation
├── data/                      # Main data directory
│   ├── economic/              # Macroecomic indicators
│   ├── financial/             # Microecomic indicators
│   ├── macro_micro/           # Processed Macro+Micro data
│   ├── sentiment/             # Processed sentiment data
│   └── stock/                 # Historical stock data
├── macro+micro_regression/    # Macroeconomic and microeconomic regression analysis
│   ├── cdnod/                 # CDNOD implementation
|      ├── cdnod_graph/        # CDNOD graphs and selected features
└── requirements.txt         # Project dependencies
```

## Setup Instructions
1. Clone this repository:
```bash
git clone https://github.com/VivianZhao12/CAPSTONE-stockreturn.git
cd CAPSTONE-stockreturn
```

2. Create and activate a new Python environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages and casaul-learn package:
```bash
pip install -r requirements.txt
pip install git+https://github.com/py-why/causal-learn.git

# Install Graphviz system package for causal graph visulization
For macOS:
brew install graphviz

For Ubuntu/Debian:
sudo apt-get install graphviz

For Windows:
choco install graphviz

Verify with:
dot -V
```


## Data Preparation and Model Training
1. Economic Impact Analysis
```bash
cd macro+micro_regression/cdnod

# Fetch all data from api and run cdnod
visit "https://www.alphavantage.co/support/#api-key" and generate your own token, replace api_key = "" with your token in cdnod.py
python cdnod.py

# Align frequency for macro and micro data
cd ../
python align_frequency_test.py

# Create data with cdnod selected features
python cdnod/create_df_cdnod.py

Note: We have already pre-selected features based on our resulting causal graph.
For future iterations, follow these steps to select new features:
1. run "python cdnod/cdnod_feature_selection.py" to automatically select the feature, results are in causal_feature.json under /cdnod_graph
2. Read the resulting casual graph and add more features if not captured
3. Modify the "features" variable in create_df_cdnod.py to include your updated feature lists!
```

2. DeepAR Model:
```bash
cd ../DeepAR

# Prepare the data
## to run with sentiment data
python preprocess.py <ticker_in_lowercase> --with_sentiment

## to run without sentiment data
python preprocess.py <ticker_in_lowercase>

# Train the model
python train.py --ticker <ticker_in_lowercase>

# Evaluate the model
python evaluate.py --ticker <ticker_in_lowercase>

## to run with sentiment data
python load_model_results.py <ticker_in_lowercase> --with_sentiment

## to run without sentiment data
python load_model_results.py <ticker_in_lowercase>
```

3. Sentiment Analysis
  ```bash
# Scraping and interpolating data from Google and CSV
python data/sentiment/Google_and_CSV_data_sraping.ipynb

python data/sentiment/Sentiment_score_with_interpolation.ipynb
```
