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

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Data Preparation and Model Training
1. DeepAR Model:
```bash
# Prepare the data
python DeepAR/preprocess.py

# Train the model
python DeepAR/train.py --config config.yml
```

2. Sentiment Analysis
  ```bash
# Scraping and interpolating data from Google and CSV
python data/sentiment/Google_and_CSV_data_sraping.ipynb

python data/sentiment/Sentiment_score_with_interpolation.ipynb
```

4. Economic Impact Analysis (In Development)

