Small ML project with stock data

## Stages
1. Define most effective metrics - DONE
2. Create derivatives - DONE
3. Train model - DONE
4. Test model with examples

## Project Structure

| File | Purpose |
|------|---------|
| `fetchData.py` | Downloads S&P 500 daily OHLCV history to `sp500_history.csv` |
| `dataCoefficent.py` | Builds features for top 100 stocks, computes Spearman correlations |
| `preprocess.py` | Feature engineering, normalization, 30-day sliding windows, chronological train/val/test split |
| `model.py` | PyTorch Dataset, LSTM+Transformer hybrid model, training loop with early stopping |
| `displayData.py` | Data visualization |

## How to Run

```bash
# 1. Fetch data (takes a while, downloads all S&P 500 history)
python fetchData.py

# 2. Train the model
python model.py
```

Training outputs `model_weights.pth` which can be loaded for predictions:
```python
from model import StockPredictor
import torch

model = StockPredictor()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
```

## Model Architecture

LSTM + Transformer hybrid predicting two regression targets:
- **Close return** — next-day % change in close price
- **Dividend yield** — next-day dividend / close price

Pipeline: `Input (30 days x 11 features) → Linear projection → 2-layer LSTM → Positional encoding → 2-layer Transformer encoder (4 heads) → Regression head → 2 outputs`

### Input Features (selected via correlation analysis)
MA 50/200 Ratio, Momentum 30d, Momentum 10d, RSI 14, Daily Return, Volatility 20d, Volume Ratio, High-Low Spread, Close-Open Spread, Upper Shadow, Lower Shadow

### Training Configuration
- Loss: MSE (regression)
- Batch size: 64
- Optimizer: Adam with weight decay
- LR scheduler: ReduceLROnPlateau
- Early stopping: patience 10
- Gradient clipping: max norm 1.0

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

## How to Improve Model Accuracy

### Model improvements
- **Weighted loss** — dividend yield is mostly zeros (non-dividend days). Use a weighted MSE that penalizes dividend errors more heavily, or train separate heads with separate losses.
- **Larger model** — increase `d_model` (64 → 128 or 256), add more transformer layers, or add more LSTM layers. Only do this if validation loss is still decreasing when early stopping triggers.
- **Attention masking** — add causal masking to the transformer so it can only attend to past timesteps, matching the real-world constraint.
- **Ensemble** — train multiple models with different random seeds or hyperparameters and average their predictions. Simple ensembling often gives 5-15% error reduction.

### Training improvements
- **Learning rate warmup** — use a linear warmup for the first few epochs before the cosine/plateau schedule. Transformers are sensitive to high initial learning rates.
- **Huber loss** — replace MSE with `nn.SmoothL1Loss()`. It's less sensitive to outlier returns (earnings days, crashes) which can dominate MSE.
- **Walk-forward validation** — instead of a single train/val/test split, use rolling windows: train on months 1-12, validate on 13, slide forward. This tests robustness across market regimes.
- **GPU training** — install PyTorch with CUDA support for 10-50x faster training, allowing more hyperparameter experiments.
- **Hyperparameter search** — use Optuna or Ray Tune to systematically search learning rate, dropout, d_model, window size, and number of layers.
