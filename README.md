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
│   ├── experiments/           # Per-company experimental results
│   └── README.md              # DeepAR-specific documentation
├── FinBERT/                   # FinBERT sentiment analysis implementation
├── data/                      # Data directory
│   ├── economic/              # Macroeconomic indicators
│   ├── financial/             # Microeconomic indicators
│   ├── macro_micro/           # Processed macro and micro data
│   ├── sentiment/             # Processed sentiment data
│   └── stock/                 # Historical stock price data
├── macro+micro_regression/    # Economic regression analysis
│   ├── cdnod/                 # CDNOD implementation
│      ├── cdnod_graph/        # CDNOD graphs and selected features
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

**Note**: Pre-selected features based on causal analysis are included. For custom feature selection:
1. Run `python cdnod/cdnod_feature_selection.py` to generate causal graphs
2. Review results in `/cdnod_graph/causal_feature.json`
3. Update feature list in `create_df_cdnod.py` as needed


### 2. Sentiment Analysis

```bash
# Process Google and CSV data sources:
python data/sentiment/Google_and_CSV_data_sraping.ipynb

# Generate interpolated sentiment scores:
python data/sentiment/Sentiment_score_with_interpolation.ipynb
```

### 2. Time Series Modeling with DeepAR

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

[4] Zhang, Y., Jiang, Q., Li, S., Chen, X., Zhang, Y., Wu, X., & Cai, M. (2019). You May Not Need Order in Time Series Forecasting. arXiv preprint arXiv:1910.09620.