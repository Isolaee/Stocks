Small ML project with stock data

## Stages
1. Define most effective metrics - DONE
2. Create derivatives - DONE
3. Train model - DONE (local + GCP Vertex AI)
4. Hyperparameter tuning - IN PROGRESS
5. Test model with examples

## Project Structure

| File | Purpose |
|------|---------|
| `fetchData.py` | Downloads S&P 500 daily OHLCV history to `sp500_history.csv` |
| `dataCoefficent.py` | Builds features for top 100 stocks, computes Spearman correlations |
| `preprocess.py` | Feature engineering, normalization, 90-day sliding windows, chronological train/val/test split |
| `model.py` | PyTorch Dataset, LSTM+Transformer hybrid model, training loop with early stopping |
| `train.py` | Vertex AI training entrypoint — reads all config from env vars |
| `gcs.py` | GCS helpers: download inputs, upload checkpoints and model weights |
| `Dockerfile` | Container image for Vertex AI custom training jobs |
| `cloudbuild.yaml` | Cloud Build pipeline: build image → push to Artifact Registry → submit Vertex AI job |
| `displayData.py` | Data visualization |

## How to Run Locally

```bash
# 1. Fetch data (takes a while, downloads all S&P 500 history)
python fetchData.py

# 2. Train the model
python model.py
```

## How to Train on GCP (Vertex AI)

### Prerequisites (run once)
```bash
# Create Artifact Registry repo
gcloud artifacts repositories create stocks-nn --repository-format=docker --location=europe-west4 --description="Stocks NN training images"

# Grant Cloud Build service account permissions
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/logging.logWriter"
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/aiplatform.user"
```

### Submit a training run
```bash
gcloud beta builds submit --region=europe-west4 --config cloudbuild.yaml --substitutions _GCS_BUCKET=stocks-nn-project-2b08a31c,_RUN_ID=v1
```

Each run builds the Docker image, pushes it to Artifact Registry, and submits a Vertex AI custom job. Model weights and checkpoints are saved to GCS at `gs://BUCKET/output/RUN_ID/`.

### Monitor training
```bash
# Check job status
gcloud ai custom-jobs list --region=europe-west4

# Stream logs for a specific job
gcloud ai custom-jobs stream-logs JOB_ID --region=europe-west4
```

### Download trained model
```bash
gsutil cp gs://stocks-nn-project-2b08a31c/output/v1/model_weights.pth .
```

### Load model for inference
```python
from model import StockPredictor
import torch

model = StockPredictor()
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()
```

### Configurable training parameters
Edit the env vars in `cloudbuild.yaml` before submitting, or pass via `--substitutions`:

| Env var | Default | Description |
|---|---|---|
| `EPOCHS` | 100 | Max training epochs |
| `BATCH_SIZE` | 64 | Samples per gradient update |
| `LR` | 0.001 | Initial learning rate |
| `PATIENCE` | 10 | Early stopping patience |
| `CHUNKED` | 1 | Use memory-efficient chunked preprocessing |
| `RESUME` | 0 | Resume from latest GCS checkpoint |

### GPU quota
By default the job runs on CPU (`n1-standard-8`). To enable GPU training, request quota at:
**GCP Console → IAM & Admin → Quotas & System Limits → Vertex AI API → Custom model training Nvidia T4 GPUs → region: europe-west4**

Then add back to `cloudbuild.yaml` under `machineSpec`:
```yaml
acceleratorType: NVIDIA_TESLA_T4
acceleratorCount: 1
```

---

## Hyperparameter Tuning on GCP (Vertex AI)

Vertex AI has a built-in hyperparameter tuning service that runs parallel trials automatically using Bayesian optimization.

### Setup

1. Add `cloudml-hypertune` to `requirements-cloud.txt`:
```
cloudml-hypertune
```

2. Report the metric at the end of `train.py`:
```python
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="val_loss",
    metric_value=best_val_loss,
    global_step=epoch,
)
```

3. Create a `hptuning.yaml` search space config:
```yaml
studySpec:
  metrics:
    - metricId: val_loss
      goal: MINIMIZE
  parameters:
    - parameterId: LR
      doubleValueSpec:
        minValue: 0.0001
        maxValue: 0.01
    - parameterId: BATCH_SIZE
      discreteValueSpec:
        values: [32, 64, 128]
    - parameterId: PATIENCE
      integerValueSpec:
        minValue: 5
        maxValue: 20
  algorithm: ALGORITHM_UNSPECIFIED  # Bayesian optimization
maxTrialCount: 20
parallelTrialCount: 4
```

4. Submit the tuning job:
```bash
gcloud ai hp-tuning-jobs create --region=europe-west4 --display-name=stocks-nn-hptuning --config=hptuning.yaml --max-trial-count=20 --parallel-trial-count=4
```

5. Check results:
```bash
gcloud ai hp-tuning-jobs list --region=europe-west4
```

The best trial's hyperparameters are shown in the Console under **Vertex AI → Training → Hyperparameter Tuning Jobs**.

---

## Model Architecture

LSTM + Transformer hybrid predicting two regression targets over a 6-month horizon:
- **Close return** — 6-month forward % change in close price
- **Dividend yield** — 6-month cumulative dividend / close price

Pipeline: `Input (90 days x 21 features) → Linear projection → 2-layer LSTM → Positional encoding → 2-layer Transformer encoder (4 heads) → Regression head → 2 outputs`

### Input Features (selected via correlation analysis)
Technical: MA 50/200 Ratio, Momentum 30d, Momentum 10d, RSI 14, Daily Return, Volatility 20d, Volume Ratio, High-Low Spread, Close-Open Spread, Upper Shadow, Lower Shadow, MACD, MACD Signal, Bollinger %B, Bollinger Width, ATR 14, OBV Change 20d

Macro: VIX, SP500_Return, Treasury_10Y, USD_Index

### Training Configuration
- Loss: MSE (regression)
- Batch size: 64
- Optimizer: Adam with weight decay
- LR scheduler: ReduceLROnPlateau
- Early stopping: patience 10
- Gradient clipping: max norm 1.0

---

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

---

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
