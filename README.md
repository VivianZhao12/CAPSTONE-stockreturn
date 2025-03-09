# Causal Discovery Framework for Stock Return Prediction

This repository implements a comprehensive multi-model framework for stock return prediction that integrates sentiment analysis, historical stock data, and macroeconomic indicators. The framework leverages state-of-the-art methods including DeepAR for time series forecasting, FinBERT for sentiment analysis, and causal discovery techniques for feature selection.

## Framework Overview

Our integrated approach consists of three primary components:

1. **Economic Analysis**: Employing CDNOD (Causal Discovery in Non-stationary with Distributional Shifts) and Random Forest regressors to model macroeconomic and microeconomic impacts
2. **Sentiment Analysis**: Implementing FinBERT to quantify market sentiment
3. **Time Series Forecasting**: Utilizing DeepAR with PCMCI+ feature selection for causal temporal relationships

## Repository Structure

```
CAPSTONE-stockreturn/
├── DeepAR/                    # DeepAR model implementation
│   ├── data/                  # Processed stock data for model training
│   │   ├── abt_stock_processed/
│   │   ├── amgn_stock_processed/
│   │   ├── amzn_stock_processed/
│   │   ├── cvs_stock_processed/
│   │   ├── goog_stock_processed/
│   │   └── t_stock_processed/
│   ├── experiments/           # Per-company experimental results
│   │   ├── abt_base_model/
│   │   ├── amgn_base_model/
│   │   ├── amzn_base_model/
│   │   ├── cvs_base_model/
│   │   ├── goog_base_model/
│   │   ├── param_search/
│   │   └── t_base_model/
│   ├── model/                 # Model architecture definitions
│   │   ├── LSTM.py
│   │   └── net.py
│   ├── config.yml             # Configuration settings
│   ├── dataloader.py          # Data loading utilities
│   ├── deepar_prediction.py   # Long-term prediction script
│   ├── evaluate.py            # Model evaluation script
│   ├── fusion_layer.py        # Fusion model implementation
│   ├── fusion_visualization.py # Visualization utilities
│   ├── load_model_results.py  # Results analysis script
│   ├── pcmci.py               # PCMCI+ feature selection
│   ├── pcmci_result.log       # Feature selection results
│   ├── preprocess.py          # Data preprocessing script
│   ├── train.py               # Model training script
│   └── utils.py               # Utility functions
├── FinBERT/                   # FinBERT sentiment analysis implementation
├── data/                      # Data directory
│   ├── economic/              # Macroeconomic indicators
│   ├── financial/             # Microeconomic indicators
│   ├── macro_micro/           # Processed macro and micro data
│   ├── sentiment/             # Processed sentiment data
│   └── stock/                 # Historical stock price data
├── macro+micro_regression/    # Economic regression analysis
│   ├── cdnod/                 # CDNOD implementation
│   │   └── cdnod_graph/       # CDNOD graphs and selected features
│   │   |   └── cdnod.py       # Fetch all data and Run CDNOD for feature selection
│   │   |   └── cdnod_feature_selection.py       # Automatic select feature from CDNOD results 
│   │   |   └── create_df_cdnod.py               # Generate Dataframe with selected features for fusion layer prediction
│   │   └── align_frequency_test.py              # Macro, Micro data alignment
└── requirements.txt           # Project dependencies
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VivianZhao12/CAPSTONE-stockreturn.git
   cd CAPSTONE-stockreturn
   ```

2. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/py-why/causal-learn.git
   ```

4. **Install Graphviz** (required for causal graph visualization):

   - **macOS**:
     ```bash
     brew install graphviz
     ```
     
   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install graphviz
     ```
     
   - **Windows**:
     ```bash
     choco install graphviz
     ```
     
   Verify installation:
   ```bash
   dot -V
   ```

## Implementation Guide

### 1. Economic Causal Analysis

```bash
cd macro+micro_regression/cdnod

# Fetch and process data
# Note: Obtain API key from "https://www.alphavantage.co/support/#api-key"
# Replace api_key = "" with your token in cdnod.py
python cdnod.py

# Align macro and micro data frequencies
cd ../
python align_frequency_test.py

# Generate dataset with CDNOD-selected features
python cdnod/create_df_cdnod.py
```

**Note**: Pre-selected features based on current causal graph are defined. For updated feature list based on current iteration:
1. Run `python cdnod/cdnod_feature_selection.py` to automatically capture important features from resulting cdnod graphs
2. Get selected feature lists in `/cdnod_graph/causal_feature.json` 
3. Update variable named "feature" in `create_df_cdnod.py` with your new feature list


### 2. Sentiment Analysis

```bash
# This part may take several hours to run. The data already exists, you may proceed with the rest of the part using the existing data.
# Please set up your own Reddit API on this platform: https://www.reddit.com/prefs/apps, including: client_id,  client_secret, username, and password.
cd data/sentiment
python sentiment_data_collection_preprocess.py 
```

### 3. Time Series Modeling with DeepAR

```bash
cd ../DeepAR
```

#### Causal Feature Selection:
```bash
# Automatic feature selection with PCMCI+
python pcmci.py
```

Pre-selected features are included. For custom selection:
1. Review results in `pcmci_result.log`
2. Update features in `preprocess.py`'s `gen_covariates` function

#### Model Training Pipeline:

**Data Preparation**:
```bash
# With sentiment analysis:
python preprocess.py <ticker_in_lowercase> --with_sentiment

# Without sentiment analysis:
python preprocess.py <ticker_in_lowercase>
```

**Model Training**:
```bash
python train.py --ticker <ticker_in_lowercase>
```

**Model Evaluation**:
```bash
python evaluate.py --ticker <ticker_in_lowercase>
```

**Analyze Model Performance Across Epochs**:
```bash
# With sentiment:
python load_model_results.py <ticker_in_lowercase> --with_sentiment

# Without sentiment:
python load_model_results.py <ticker_in_lowercase>
```

**Generate Long-term Predictions**:
```bash
# With sentiment:
python deepar_prediction.py <ticker_in_lowercase> --with-sentiment --epoch <epoch_number>

# Without sentiment:
python deepar_prediction.py <ticker_in_lowercase> --epoch <epoch_number>
```

**Fusion Layer for Final Predictions**:
```bash
python fusion_layer.py <ticker_in_lowercase>
```

**Visualization**:
```bash
# With sentiment:
python fusion_visualization.py <ticker_in_lowercase> --with-sentiment

# Without sentiment:
python fusion_visualization.py <ticker_in_lowercase>
```

## References
[1] Huang, A. H., Wang, H., & Yang, Y. (2022). FinBERT: A Large Language Model for Extracting Information from Financial Text. Contemporary Accounting Research, 39(4), 2979-3000. https://doi.org/10.1111/1911-3846.12832

[2] Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181-1191. https://doi.org/10.1016/j.ijforecast.2019.07.001

[3] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances, 5(11), eaau4996. https://doi.org/10.1126/sciadv.aau4996 

[4] Zhang, Y., Jiang, Q., Li, S., Jin, X., Ma, X., & Yan, X. (2019). You May Not Need Order in Time Series Forecasting. arXiv preprint arXiv:1910.09620. https://doi.org/10.48550/arXiv.1910.09620
