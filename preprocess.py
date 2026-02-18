import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Features selected based on correlation analysis (corr_close_avg.csv)
# Excluded: Cumulative Return (derived from target), Open/High/Low (too correlated with Close)
FEATURE_COLS = [
    # Per-stock technical
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
    "MACD",
    "MACD Signal",
    "Bollinger %B",
    "Bollinger Width",
    "ATR 14",
    "OBV Change 20d",
    # External macro (merged by date)
    "VIX",
    "SP500_Return",
    "Treasury_10Y",
    "USD_Index",
]

MACRO_COLS = ["VIX", "SP500_Return", "Treasury_10Y", "USD_Index"]

WINDOW_SIZE = 90


def load_macro(macro_path="macro_history.csv"):
    """Load macro indicator CSV (VIX, SP500_Return, Treasury_10Y, USD_Index)."""
    macro = pd.read_csv(macro_path)
    macro["Date"] = pd.to_datetime(macro["Date"], utc=True)
    macro = macro.set_index("Date")
    # Ensure no gaps — forward-fill missing trading days
    macro = macro.ffill()
    return macro


def load_and_enrich(csv_path="sp500_history.csv", macro_path="macro_history.csv"):
    """Load raw CSV, add derived features per stock, and merge macro data."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    macro = load_macro(macro_path)
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

        # MACD (EMA12 - EMA26) and Signal line (EMA9 of MACD)
        ema12 = g["Close"].ewm(span=12).mean()
        ema26 = g["Close"].ewm(span=26).mean()
        g["MACD"] = ema12 - ema26
        g["MACD Signal"] = g["MACD"].ewm(span=9).mean()

        # Bollinger Bands (20-day)
        bb_ma = g["Close"].rolling(20).mean()
        bb_std = g["Close"].rolling(20).std()
        g["Bollinger %B"] = (g["Close"] - (bb_ma - 2 * bb_std)) / (4 * bb_std)
        g["Bollinger Width"] = (4 * bb_std) / bb_ma

        # ATR 14 (Average True Range)
        tr = pd.concat([
            g["High"] - g["Low"],
            (g["High"] - g["Close"].shift(1)).abs(),
            (g["Low"] - g["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        g["ATR 14"] = tr.rolling(14).mean()

        # OBV % change (20-day rate of change of On-Balance Volume)
        signed_vol = g["Volume"].where(g["Close"].diff() > 0, -g["Volume"])
        signed_vol = signed_vol.where(g["Close"].diff() != 0, 0)
        obv = signed_vol.cumsum()
        g["OBV Change 20d"] = obv.pct_change(20)

        # Regression targets — 6-month (126 trading days) forward-looking
        # Target 1: 6-month forward close % return
        g["Target_Close"] = g["Close"].shift(-126) / g["Close"] - 1
        # Target 2: 6-month cumulative dividend yield (sum of next 126 days' dividends / current close)
        g["Target_Dividend"] = (
            g["Dividends"].rolling(126, min_periods=1).sum().shift(-126) / g["Close"]
        ).fillna(0.0)

        enriched.append(g)

    df = pd.concat(enriched, ignore_index=True)

    # Merge macro indicators by date
    df = df.merge(macro, left_on="Date", right_index=True, how="left")
    for col in MACRO_COLS:
        df[col] = df[col].ffill()

    return df


def normalize_features(df, feature_cols=FEATURE_COLS, macro_cols=MACRO_COLS):
    """Normalize features: per-stock for technical indicators, globally for macro.

    Returns the DataFrame with normalized feature columns and
    a dict of {symbol: fitted_scaler, "__macro__": macro_scaler} for inverse transforms.
    """
    scalers = {}
    stock_cols = [c for c in feature_cols if c not in macro_cols]

    # Per-stock normalization for technical indicators
    for symbol, group in df.groupby("Symbol"):
        subset = group[stock_cols]
        # Skip stocks where any feature column is entirely NaN
        if subset.isna().all().any():
            df = df.drop(group.index)
            continue
        scaler = MinMaxScaler()
        df.loc[group.index, stock_cols] = scaler.fit_transform(subset)
        scalers[symbol] = scaler

    # Global normalization for macro features (same values across all stocks on a given date)
    present_macro = [c for c in macro_cols if c in df.columns]
    if present_macro:
        macro_scaler = MinMaxScaler()
        df[present_macro] = macro_scaler.fit_transform(df[present_macro])
        scalers["__macro__"] = macro_scaler

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


def prepare_data(csv_path="sp500_history.csv", macro_path="macro_history.csv"):
    """Full pipeline: load → enrich → merge macro → normalize → window → split.

    Returns dict with train/val/test splits, each containing (X, y, meta).
    """
    print("Loading and enriching data...")
    df = load_and_enrich(csv_path, macro_path)

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
