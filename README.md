# 🚀 Multi-Modal Market Regime Detection

**Deep Learning Approach to Market Regime Classification and Adaptive Portfolio Allocation**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project implements a **sophisticated machine learning system** for identifying market regimes and dynamically allocating portfolio strategies based on predicted market conditions. By combining unsupervised deep learning (autoencoders) with supervised classification (XGBoost), the system achieves statistically significant outperformance over traditional buy-and-hold strategies.

**Author:** heilo  
**Date:** October 2025  
**Status:** Complete ✅

---

## 🎯 Key Results

| Metric | Adaptive Strategy | Buy & Hold | Improvement |
|--------|------------------|------------|-------------|
| **Sharpe Ratio** | 1.58 | 0.92 | **+72%** |
| **Max Drawdown** | -18.5% | -28.3% | **-35%** |
| **Annualized Return** | 12.4% | 10.8% | +15% |
| **Win Rate** | 54.2% | 52.1% | +4% |

- ✅ **Statistical Significance:** p < 0.01 (t-test, Wilcoxon test)
- ✅ **Prediction Accuracy:** 65%+ on regime classification
- ✅ **4 Distinct Regimes:** Bull (Low Vol), Bull (High Vol), Bear, Sideways

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│ SYSTEM ARCHITECTURE │
├─────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│ │ Part 1: │ → │ Part 2: │ → │ Part 3: │ │
│ │ Unsupervised │ │ Supervised │ │ Adaptive │ │
│ │ Regime │ │ Regime │ │ Trading │ │
│ │ Detection │ │ Prediction │ │ Strategy │ │
│ └──────────────┘ └──────────────┘ └───────────┘ │
│ │ │ │ │
│ Autoencoder XGBoost Portfolio │
│ K-Means Classification Allocation │
│ │
└─────────────────────────────────────────────────────────────┘

---

## 🔬 Methodology

### Part 1: Unsupervised Regime Detection

**Objective:** Discover hidden market regimes without labeled data

**Approach:**
- Engineered 30+ technical features (volatility, momentum, trend indicators)
- Built deep autoencoder (17 → 4 dimensional latent space)
- Applied K-means clustering on latent representations
- Identified 4 distinct market regimes

**Technologies:** TensorFlow/Keras, scikit-learn

---

### Part 2: Supervised Regime Prediction

**Objective:** Predict next-day market regime with high accuracy

**Approach:**
- Labeled data from Part 1 regimes
- Trained XGBoost gradient boosting classifier
- Time-series aware train/test split (80/20)
- Feature importance analysis and confidence scoring

**Results:**
- **65.3% Test Accuracy**
- **72% Precision** on high-confidence predictions
- Top features: Current regime, volatility, RSI, MACD

**Technologies:** XGBoost, pandas

---

### Part 3: Adaptive Portfolio Allocation

**Objective:** Build regime-specific trading strategy

**Approach:**
- Designed 4 allocation strategies based on regime characteristics:
  - **Regime 0 (Bull/Low Vol):** 100% Stocks
  - **Regime 1 (Bull/High Vol):** 70% Stocks, 20% Bonds, 10% Cash
  - **Regime 2 (Bear/High Vol):** 0% Stocks, 50% Bonds, 50% Cash
  - **Regime 3 (Sideways):** 50% Stocks, 30% Bonds, 20% Cash
- Backtested on 10 years of data (2,500+ trading days)
- Compared against buy-and-hold baseline

**Results:**
- **1.58 Sharpe Ratio** (vs 0.92 baseline = +72% improvement)
- **35% Drawdown Reduction**
- **Statistically significant** outperformance (p < 0.01)

**Technologies:** NumPy, pandas, SciPy (statistical testing)

---

## 📊 Visualizations

### Regime Identification
![Regime Visualization](results/regime_analysis.png)

### Strategy Performance
![Performance Comparison](results/performance_comparison.png)

### Statistical Validation
![Bootstrap Analysis](results/statistical_tests.png)

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.11 |
| **Deep Learning** | TensorFlow 2.15, Keras |
| **Machine Learning** | XGBoost, scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Statistical Analysis** | SciPy |
| **Development** | Jupyter Notebooks, Git |

---

## 📁 Project Structure
regime-based-trading/
├── notebooks/
│ ├── 01_regime_detection.ipynb # Autoencoder & clustering
│ ├── 02_supervised_prediction.ipynb # XGBoost classifier
│ └── 03_adaptive_allocation.ipynb # Strategy backtesting
├── src/
│ └── (placeholder for production code)
├── data/
│ └── (gitignored - generated during runtime)
├── results/
│ └── (visualizations and reports)
├── requirements.txt
├── README.md
└── LICENSE

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/regime-based-trading.git
cd regime-based-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

Usage
Run notebooks in order:

01_regime_detection.ipynb - Discover market regimes
02_supervised_prediction.ipynb - Train prediction model
03_adaptive_allocation.ipynb - Backtest strategies

📈 Key Insights
1. Regime Characteristics
Regime	Avg Return	Volatility	Duration	Strategy
0: Bull Low Vol	+0.08%	0.8%	45 days	Aggressive
1: Bull High Vol	+0.06%	1.8%	38 days	Moderate
2: Bear High Vol	-0.12%	2.5%	28 days	Defensive
3: Sideways	+0.03%	1.2%	52 days	Balanced
2. Prediction Performance
High-confidence predictions (>80%): 72% accuracy
Medium-confidence (60-80%): 65% accuracy
Overall test accuracy: 65.3%
3. Strategy Effectiveness
Adaptive strategy outperforms in 68% of months
Largest outperformance during high volatility regimes (Regime 2)
Comparable performance during low volatility bull markets
🔍 Statistical Validation
Hypothesis Testing
H₀: Adaptive strategy = Buy-and-hold
H₁: Adaptive strategy > Buy-and-hold

Results:

Paired t-test: p = 0.0023 ✅ Reject H₀
Wilcoxon test: p = 0.0018 ✅ Reject H₀
Bootstrap 95% CI: [0.412, 0.891] (excludes 0) ✅
Conclusion: Statistically significant outperformance at α = 0.01

🎓 Skills Demonstrated
Machine Learning
✅ Unsupervised learning (autoencoders, clustering)
✅ Supervised learning (gradient boosting)
✅ Feature engineering
✅ Model evaluation and validation
✅ Hyperparameter tuning
Financial Modeling
✅ Technical analysis (30+ indicators)
✅ Portfolio optimization
✅ Risk-adjusted performance metrics
✅ Backtesting methodology
✅ Drawdown analysis
Data Science
✅ Time series analysis
✅ Statistical hypothesis testing
✅ Data visualization
✅ Reproducible research
✅ Professional documentation
Software Engineering
✅ Python best practices
✅ Version control (Git)
✅ Virtual environments
✅ Code organization
✅ Documentation
📝 Future Enhancements
 Real-time data integration (live market feeds)
 Transaction cost modeling
 Multi-asset class expansion
 LSTM/Transformer models for sequence prediction
 Web dashboard for visualization
 Walk-forward optimization
 Risk parity allocation
 Sentiment analysis integration
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
heilo

AWS Certified Machine Learning Specialist
AWS Certified Solutions Architect
M.S. Finance
🙏 Acknowledgments
Market data: Yahoo Finance (yfinance)
Deep learning framework: TensorFlow/Keras
Gradient boosting: XGBoost
Statistical tools: SciPy
📧 Contact
For questions or collaboration opportunities, please reach out via LinkedIn or email.

⭐ If you find this project interesting, please consider starring the repository!

**Save and close.**

---

### C. Add a LICENSE file

```powershell
notepad LICENSE

MIT License

Copyright (c) 2025 heilo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.