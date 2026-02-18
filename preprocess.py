import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Features selected based on correlation analysis (corr_close_avg.csv)
# Excluded: Cumulative Return (derived from target), Open/High/Low (too correlated with Close)
FEATURE_COLS = [
    "MA 50/200 Ratio",
    "Momentum 30d",
    "Momentum 10d",
    "RSI 14",
    "Daily Return",
    "Volatility 20d",
    "Volume Ratio",
    "High-Low Spread",
    "Close-Open Spread",
    "Upper Shadow",
    "Lower Shadow",
]

WINDOW_SIZE = 30


def load_and_enrich(csv_path="sp500_history.csv"):
    """Load raw CSV and add derived features per stock."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    enriched = []

    for symbol, group in df.groupby("Symbol"):
        g = group.sort_values("Date").copy()

        # Derived features (same logic as dataCoefficent.py)
        g["Daily Return"] = g["Close"].pct_change()
        g["Volatility 20d"] = g["Daily Return"].rolling(20).std()
        g["MA 50"] = g["Close"].rolling(50).mean()
        g["MA 200"] = g["Close"].rolling(200).mean()
        g["MA 50/200 Ratio"] = g["MA 50"] / g["MA 200"]
        g["Volume MA 20"] = g["Volume"].rolling(20).mean()
        g["Volume Ratio"] = g["Volume"] / g["Volume MA 20"]
        g["High-Low Spread"] = g["High"] - g["Low"]
        g["Close-Open Spread"] = g["Close"] - g["Open"]
        g["Upper Shadow"] = g["High"] - g[["Close", "Open"]].max(axis=1)
        g["Lower Shadow"] = g[["Close", "Open"]].min(axis=1) - g["Low"]
        g["Momentum 10d"] = g["Close"] - g["Close"].shift(10)
        g["Momentum 30d"] = g["Close"] - g["Close"].shift(30)
        g["RSI 14"] = g["Daily Return"].rolling(14).apply(
            lambda x: (
                100 - 100 / (1 + x[x > 0].mean() / abs(x[x < 0].mean()))
                if abs(x[x < 0].mean()) > 0
                else 100
            ),
            raw=False,
        )

        # Regression targets
        # Target 1: next-day close % return
        g["Target_Close"] = g["Close"].pct_change().shift(-1)
        # Target 2: next-day dividend yield (dividend / close price, 0 on non-dividend days)
        g["Target_Dividend"] = (g["Dividends"].shift(-1) / g["Close"]).fillna(0.0)

        enriched.append(g)

    return pd.concat(enriched, ignore_index=True)


def normalize_features(df, feature_cols=FEATURE_COLS):
    """Normalize features per stock using MinMaxScaler.

    Returns the DataFrame with normalized feature columns and
    a dict of {symbol: fitted_scaler} for inverse transforms later.
    """
    scalers = {}

    for symbol, group in df.groupby("Symbol"):
        subset = group[feature_cols]
        # Skip stocks where any feature column is entirely NaN
        if subset.isna().all().any():
            df = df.drop(group.index)
            continue
        scaler = MinMaxScaler()
        df.loc[group.index, feature_cols] = scaler.fit_transform(subset)
        scalers[symbol] = scaler

    return df, scalers


TARGET_COLS = ["Target_Close", "Target_Dividend"]


def create_windows(df, window_size=WINDOW_SIZE, feature_cols=FEATURE_COLS,
                   target_cols=TARGET_COLS):
    """Create sliding windows of features + targets per stock.

    Returns:
        X: np.ndarray of shape (num_samples, window_size, num_features)
        y: np.ndarray of shape (num_samples, 2) — [close_return, dividend_yield]
        meta: list of dicts with symbol and date for each sample
    """
    all_X, all_y, meta = [], [], []

    for symbol, group in df.groupby("Symbol"):
        g = group.sort_values("Date").dropna(subset=feature_cols + target_cols)
        features = g[feature_cols].values
        targets = g[target_cols].values
        dates = g["Date"].values

        for i in range(len(g) - window_size):
            all_X.append(features[i : i + window_size])
            all_y.append(targets[i + window_size - 1])
            meta.append({"symbol": symbol, "date": dates[i + window_size - 1]})

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32), meta


def train_val_test_split(X, y, meta, train_ratio=0.7, val_ratio=0.15):
    """Chronological split — no shuffling.

    Sorts all samples by date, then splits into train/val/test.
    """
    # Sort by date
    dates = np.array([m["date"] for m in meta])
    order = np.argsort(dates)
    X, y = X[order], y[order]
    meta = [meta[i] for i in order]

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X[:train_end], y[:train_end], meta[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end], meta[train_end:val_end]),
        "test": (X[val_end:], y[val_end:], meta[val_end:]),
    }

    for name, (Xs, ys, ms) in splits.items():
        date_range = f"{ms[0]['date']} to {ms[-1]['date']}" if ms else "empty"
        print(f"  {name}: {len(Xs):>8,} samples  ({date_range})")

    return splits


def prepare_data(csv_path="sp500_history.csv"):
    """Full pipeline: load → enrich → normalize → window → split.

    Returns dict with train/val/test splits, each containing (X, y, meta).
    """
    print("Loading and enriching data...")
    df = load_and_enrich(csv_path)

    print("Normalizing features...")
    df, scalers = normalize_features(df)

    print("Creating sliding windows...")
    X, y, meta = create_windows(df)
    print(f"  Total samples: {len(X):,}  |  Shape: {X.shape}")

    print("Splitting chronologically (70/15/15)...")
    splits = train_val_test_split(X, y, meta)

    return splits, scalers


if __name__ == "__main__":
    splits, scalers = prepare_data()
    X_train, y_train, _ = splits["train"]
    print(f"\nReady for training:")
    print(f"  X_train: {X_train.shape}  (samples, window, features)")
    print(f"  y_train: {y_train.shape}  (close_return, dividend_yield)")
    print(f"  Close return — mean: {y_train[:, 0].mean():.6f}, std: {y_train[:, 0].std():.6f}")
    print(f"  Dividend yield — mean: {y_train[:, 1].mean():.6f}, std: {y_train[:, 1].std():.6f}")
