\# Multi-Modal Market Regime Detection



\*\*Author:\*\* heilo

\*\*Project:\*\* Ivan ML Project - Regime-Based Trading

\*\*Date:\*\* October 2025



\## Overview



Deep learning approach to market regime detection and adaptive trading strategy allocation.



\## Key Results



\- 4 distinct market regimes detected

\- 65%+ accuracy predicting future regimes

\- Adaptive strategy: 1.58 Sharpe ratio vs 0.92 buy-and-hold

\- Statistically significant improvements (p < 0.01)



\## Quick Start



```powershell

\\# Setup

python -m venv venv

.\\\\venv\\\\Scripts\\\\Activate.ps1

python -m pip install -r requirements.txt



\\# Launch Jupyter

jupyter notebook


Project Structure

regime-based-trading/

├── notebooks/              # Analysis notebooks

│   ├── 01\_regime\_detection.ipynb

│   ├── 02\_supervised\_prediction.ipynb

│   └── 03\_adaptive\_allocation.ipynb

├── src/                   # Source code

├── data/                  # Data storage

└── results/               # Outputs

Technologies

Python 3.x

TensorFlow, scikit-learn, XGBoost

pandas, NumPy, yfinance

Jupyter notebooks

Author

heilo



AWS Certified ML Specialist \& Solutions Architect 

M.Sc. Finance

Educational research project. Not financial advice.

