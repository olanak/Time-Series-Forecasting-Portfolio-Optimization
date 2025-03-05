markdown
# Time Series Forecasting for Portfolio Management Optimization

[![10 Academy AI Mastery Program](https://img.shields.io/badge/10%20Academy-AI%20Mastery-blue)](https://www.10academy.org)

This repository contains code and documentation for a project applying time series forecasting to optimize investment portfolios. It aligns with the **10 Academy Artificial Intelligence Mastery Program - Week 11 Challenge**, focusing on leveraging historical financial data to predict market trends and enhance portfolio performance.

---

## Table of Contents

- [Business Objective](#business-objective)
- [Data Overview](#data-overview)
- [Expected Outcomes](#expected-outcomes)
- [Tasks](#tasks)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Team](#team)
- [References](#references)

---

## Business Objective

**Guide Me in Finance (GMF) Investments** is a financial advisory firm specializing in personalized portfolio management. The goal is to use advanced time series forecasting models to:  
- Predict market trends.  
- Optimize asset allocation.  
- Enhance portfolio performance.  

Financial analysts at GMF analyze real-time financial data (e.g., from YFinance) to provide actionable insights, ensuring strategies are based on the latest market conditions. The project aims to minimize risks and maximize returns for clients through data-driven decision-making.

---

## Data Overview

### Dataset Details  
Historical financial data for three assets is used:  
1. **Tesla (TSLA)**: High-growth, high-risk stock in the consumer discretionary sector.  
2. **Vanguard Total Bond Market ETF (BND)**: A stable, low-risk bond ETF.  
3. **S&P 500 ETF (SPY)**: A diversified, moderate-risk index fund.  

#### Data Source  
- Extracted via the `yfinance` Python library.  
- Timeframe: **January 1, 2015, to January 31, 2025**.  

#### Features  
- **Date**: Trading day timestamp.  
- **Open, High, Low, Close**: Daily price metrics.  
- **Adj Close**: Adjusted close price (accounts for dividends/splits).  
- **Volume**: Total shares/units traded daily.  

---

## Expected Outcomes  

### Skills  
- Competence in time series forecasting (ARIMA, SARIMA, LSTM).  
- Experience with financial data preprocessing and analysis using `yfinance`.  
- Ability to develop, evaluate, and deploy predictive models.  
- Skills in portfolio optimization using forecast insights.  

### Knowledge  
- In-depth understanding of financial market trends.  
- Proficiency in data-driven portfolio management.  
- Insights into risk management and return optimization.  

---

## Tasks  

### **Task 1: Preprocess and Explore the Data**  
- **Extract Data**: Use `yfinance` to fetch historical data for TSLA, BND, and SPY.  
- **Data Cleaning**:  
  - Check basic statistics and data types.  
  - Handle missing values (fill/interpolate/remove).  
  - Normalize/scale data for machine learning models.  
- **EDA**:  
  - Visualize closing prices and daily returns.  
  - Analyze rolling means/standard deviations for volatility.  
  - Decompose time series into trend, seasonal, and residual components.  
  - Detect outliers (e.g., extreme daily returns).  

### **Task 2: Develop Time Series Forecasting Models**  
- **Choose a Model**:  
  - **ARIMA**: For univariate time series without seasonality.  
  - **SARIMA**: Extends ARIMA to account for seasonality.  
  - **LSTM**: Captures long-term dependencies in time series data.  
- **Model Training**:  
  - Split data into training/testing sets.  
  - Optimize parameters (e.g., `auto_arima` for ARIMA/SARIMA).  
- **Evaluation**:  
  - Metrics: MAE, RMSE, MAPE.  

### **Task 3: Forecast Future Market Trends**  
- **Generate Forecasts**:  
  - Predict TSLA prices for 6–12 months using the trained model.  
  - Include confidence intervals to quantify uncertainty.  
- **Analyze Results**:  
  - **Trend Analysis**: Identify long-term trends (upward/downward/stable).  
  - **Volatility**: Highlight periods of increased uncertainty.  
  - **Opportunities/Risks**: Outline potential price movements and risks.  

### **Task 4: Optimize Portfolio Based on Forecast**  
- **Portfolio Setup**:  
  - Combine forecasts for TSLA, BND, and SPY into a single DataFrame.  
- **Optimization**:  
  - Compute annual returns and covariance matrices.  
  - Use `scipy.optimize` to maximize the **Sharpe Ratio**.  
  - Adjust allocations to balance risk and return.  
- **Analysis**:  
  - Calculate expected return, volatility, and VaR.  
  - Visualize portfolio performance (e.g., cumulative returns).  

---

## Repository Structure  

```
├── notebooks/               # Jupyter notebooks for each task  
│   ├── 01_data_exploration.ipynb  
│   ├── 02_time_series_modeling.ipynb  
│   ├── 03_forecast_trends.ipynb  
│   └── 04_portfolio_optimization.ipynb  
├── data/                    # Processed and raw data  
│   ├── raw/                 # Raw data from YFinance  
│   └── processed/           # Cleaned/preprocessed data  
├── models/                  # Trained models  
│   ├── arima_model.pkl      # ARIMA model  
│   ├── sarima_model.pkl     # SARIMA model  
│   └── lstm_model.keras     # LSTM model  
├── results/                 # Visualizations and outputs  
│   ├── figures/             # Plots and charts  
│   └── forecasts.csv        # Forecasted data  
├── README.md                # This file  
└── requirements.txt         # Dependencies  
```  

---

## Installation and Setup  

### Prerequisites  
- Python 3.8+  
- Dependencies:  
  ```bash  
  pip install -r requirements.txt  
  ```  

### Key Libraries  
- `numpy`, `pandas`, `matplotlib`, `seaborn`  
- `statsmodels`, `tensorflow`, `scipy`, `sklearn`  
- `yfinance`, `pmdarima`  

---

## Team  

**Tutors**:  
- Mahlet  
- Rediet  
- Kerod  
- Elias  
- Emitinan  
- Rehmet  

---

## References  

### Time Series Forecasting  
1. [ARIMA Tutorial](https://www.datacamp.com/tutorial/arima)  
2. [ARIMA for Time Series Forecasting](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)  
3. [Time Series Analysis](https://www.geeksforgeeks.org/time-series-analysis-and-forecasting/)  

### Portfolio Optimization  
1. [Complete Guide to Portfolio Optimization](https://miltonfmr.com/the-complete-guide-to-portfolio-optimization-in-r-part1/)  
2. [Portfolio Optimization in Python](https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/)  
3. [PyPortfolioOpt GitHub](https://github.com/robertmartin8/PyPortfolioOpt)  

---

## License  

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  

--- 
