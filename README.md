Small ML project with stock data

Stages
1. Define most effective metrics - DONE
2. Create derivates
3. Train model
4. Test model with examples

## Stage 1: Correlation Analysis

Analyzed Spearman rank correlations across the top 100 US stocks by market cap to find which features correlate most strongly with Close price and Dividends on average.

### Key findings (Close price)
- MA 50/200 Ratio (+0.47) and High-Low Spread (+0.40) are the strongest non-trivial signals
- Momentum 30d (+0.32) outperforms Momentum 10d (+0.20)
- Volume and Volatility have weak negative correlation with price

### Key findings (Dividends)
- MA 200 (+0.63) is the top predictor — dividends grow with long-term price trends
- Price-level features cluster around +0.59
- Volume and momentum features show near-zero correlation

### Output files
- `corr_close_avg.csv` — averaged correlations vs Close price
- `corr_dividends_avg.csv` — averaged correlations vs Dividends
