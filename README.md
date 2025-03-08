# Causal Discovery in Stock Return

This repository contains the implementation of a multi-model framework for stock return prediction that integrates sentiment analysis, historical stock data, and macroeconomic indicators. The project uses DeepAR for time series prediction, FinBERT for sentiment analysis, and a nowcasting model for economic indicators.


## Project Overview
Our framework combines three main components:
- Time series prediction using DeepAR
- Sentiment analysis using FinBERT
- Economic impact analysis using nowcasting

## Project Structure
```bash
CAPSTONE-stockreturn/
├── DeepAR/                    # DeepAR model implementation
│   ├── experiments/           # Experiment results
│   └── README.md             # DeepAR specific documentation
├── FinBERT/                  # FinBERT implementation
├── data/                     # Main data directory
│   ├── macro_micro/         # Economic indicators
│   ├── sentiment/           # Processed sentiment data
│   └── stock/              # Historical stock data
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
For Mac
brew install graphviz

For Ubuntu/Debian
sudo apt-get install graphviz

For Windows
choco install graphviz

Verify with
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

# Predict quarterly impact with selected features for each company
python cdnod/create_df_cdnod.py

* Note we have already pre-selected the fearures based on our resulting graph, for future iterations:
To select the new feature based on the new graphs:
1. python cdnod/cdnod_feature_selection.py to automatically select the feature, result would be stored in causal_feature.json under cdnod_graph
2. Read the resulting casual graph and add more features if not captured
3. Replace "features" varaibles in "create_df_cdnod.py" with your new features!
```

2. DeepAR Model:
```bash
cd ../DeepAR
# Prepare the data
## to run with sentiment data
python preprocess.py token_lowercase --with_sentiment
## to run without sentiment data
python preprocess.py token_lowercase

* note: replace token_lowercase with company ticker in lowercase

# Train the model
python train.py
```

3. Sentiment Analysis
  ```bash
# Scraping and interpolating data from Google and CSV
python data/sentiment/Google_and_CSV_data_sraping.ipynb

python data/sentiment/Sentiment_score_with_interpolation.ipynb
```
