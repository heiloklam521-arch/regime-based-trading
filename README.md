# Multi-Modal Market Regime Detection & Adaptive Trading Strategy

## 🎯 Project Overview
A machine learning trading system that uses Hidden Markov Models (HMM) to detect market regimes and dynamically adjusts portfolio allocation across SPY, TLT, and cash positions.

## 📊 Key Results (Out-of-Sample Test Period: 482 days)
- **Sharpe Ratio**: 0.616 vs 0.563 for buy-and-hold (+9.4%)
- **Sortino Ratio**: 1.274 vs 1.089 (+17.0%)
- **Volatility**: 14.1% vs 18.7% (-24.9%)
- **Max Drawdown**: -13.1% vs -16.5% (20.6% improvement)

## 🛠️ Technical Implementation
- **Regime Detection**: 4-state Hidden Markov Model
- **Validation**: Walk-forward analysis with lagged predictions
- **Statistical Testing**: T-tests, Wilcoxon tests, 10,000-iteration bootstrap
- **Risk Management**: Regime-specific allocation strategies

## 📈 Strategy Performance
While the adaptive strategy achieved slightly lower absolute returns (20.4% vs 22.9%), it delivered superior risk-adjusted performance, particularly during neutral market conditions where it outperformed by 10.6 percentage points.

## 🔧 Technologies
- Python 3.11.9
- pandas, numpy
- scikit-learn (HMM)
- statsmodels
- matplotlib, seaborn


