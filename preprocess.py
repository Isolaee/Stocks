import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gcs import resolve, resolve_dir


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


def load_macro(macro_path=None):
    """Load macro indicator CSV (VIX, SP500_Return, Treasury_10Y, USD_Index)."""
    if macro_path is None:
        macro_path = os.environ.get("MACRO_PATH", "macro_history.csv")
    macro = pd.read_csv(resolve(macro_path))
    macro["Date"] = pd.to_datetime(macro["Date"], utc=True)
    macro = macro.set_index("Date")
    # Ensure no gaps — forward-fill missing trading days
    macro = macro.ffill()
    return macro


def load_and_enrich(csv_path=None, macro_path=None):
    """Load raw CSV, add derived features per stock, and merge macro data."""
    if csv_path is None:
        csv_path = os.environ.get("SP500_PATH", "sp500_history.csv")
    df = pd.read_csv(resolve(csv_path))
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
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
    # Replace infinities from division-by-zero in feature engineering
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

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


def prepare_data(csv_path=None, macro_path=None):
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


# ──────────────────────────────────────────────
# Chunked pipeline for large datasets
# ──────────────────────────────────────────────

def _enrich_symbol_group(group):
    """Compute derived features for a single stock group (same logic as load_and_enrich)."""
    g = group.sort_values("Date").copy()

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

    ema12 = g["Close"].ewm(span=12).mean()
    ema26 = g["Close"].ewm(span=26).mean()
    g["MACD"] = ema12 - ema26
    g["MACD Signal"] = g["MACD"].ewm(span=9).mean()

    bb_ma = g["Close"].rolling(20).mean()
    bb_std = g["Close"].rolling(20).std()
    g["Bollinger %B"] = (g["Close"] - (bb_ma - 2 * bb_std)) / (4 * bb_std)
    g["Bollinger Width"] = (4 * bb_std) / bb_ma

    tr = pd.concat([
        g["High"] - g["Low"],
        (g["High"] - g["Close"].shift(1)).abs(),
        (g["Low"] - g["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    g["ATR 14"] = tr.rolling(14).mean()

    signed_vol = g["Volume"].where(g["Close"].diff() > 0, -g["Volume"])
    signed_vol = signed_vol.where(g["Close"].diff() != 0, 0)
    obv = signed_vol.cumsum()
    g["OBV Change 20d"] = obv.pct_change(20)

    g["Target_Close"] = g["Close"].shift(-126) / g["Close"] - 1
    g["Target_Dividend"] = (
        g["Dividends"].rolling(126, min_periods=1).sum().shift(-126) / g["Close"]
    ).fillna(0.0)

    return g


def prepare_data_chunked(
    csv_path=None,
    macro_path=None,
    chunk_dir=None,
    chunk_size=50,
    train_ratio=0.7,
    val_ratio=0.15,
):
    """Memory-efficient pipeline that processes symbols in chunks and saves to disk.

    Args:
        chunk_size: number of symbols to process at a time
        chunk_dir: directory to store .npz chunk files

    Returns:
        splits: dict with train/val/test, each containing (X, y, meta)
        scalers: dict of fitted scalers
    """
    if csv_path is None:
        csv_path = os.environ.get("SP500_PATH", "sp500_history.csv")
    if macro_path is None:
        macro_path = os.environ.get("MACRO_PATH", "macro_history.csv")
    if chunk_dir is None:
        chunk_dir = os.environ.get("CHUNK_DIR", "data_chunks")

    os.makedirs(chunk_dir, exist_ok=True)

    # Load macro data once (small)
    macro = load_macro(macro_path)

    # Read only the Symbol column to get the full list without loading everything
    all_symbols = pd.read_csv(resolve(csv_path), usecols=["Symbol"])["Symbol"].unique()
    symbol_chunks = [
        all_symbols[i:i + chunk_size]
        for i in range(0, len(all_symbols), chunk_size)
    ]
    print(f"Processing {len(all_symbols)} symbols in {len(symbol_chunks)} chunks of ~{chunk_size}...")

    all_scalers = {}
    chunk_files = []

    for ci, symbols in enumerate(symbol_chunks):
        print(f"  Chunk {ci + 1}/{len(symbol_chunks)} ({len(symbols)} symbols)...")

        # Read only rows for this chunk's symbols
        df_raw = pd.read_csv(resolve(csv_path))
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], utc=True)
        df_chunk = df_raw[df_raw["Symbol"].isin(symbols)]
        del df_raw

        # Enrich each symbol
        enriched = []
        for symbol, group in df_chunk.groupby("Symbol"):
            enriched.append(_enrich_symbol_group(group))
        df_chunk = pd.concat(enriched, ignore_index=True)
        df_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Merge macro
        df_chunk = df_chunk.merge(macro, left_on="Date", right_index=True, how="left")
        for col in MACRO_COLS:
            df_chunk[col] = df_chunk[col].ffill()

        # Normalize (per-stock scalers + macro)
        df_chunk, chunk_scalers = normalize_features(df_chunk)
        all_scalers.update(chunk_scalers)

        # Create windows
        X, y, meta = create_windows(df_chunk)
        if len(X) == 0:
            continue

        # Save chunk to disk
        chunk_path = os.path.join(chunk_dir, f"chunk_{ci:03d}.npz")
        np.savez(chunk_path, X=X, y=y)
        # Save meta separately (contains Python objects)
        meta_path = os.path.join(chunk_dir, f"chunk_{ci:03d}_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        chunk_files.append((chunk_path, meta_path))

        del df_chunk, X, y, meta, enriched

    # ── Chronological split without loading all data into memory ──
    # Step 1: Collect only dates + chunk/row indices from meta files
    print("Collecting sample dates for chronological split...")
    sample_refs = []  # (date, chunk_idx, row_idx)
    chunk_sizes = []
    for ci, (chunk_path, meta_path) in enumerate(chunk_files):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        chunk_sizes.append(len(meta))
        for ri, m in enumerate(meta):
            sample_refs.append((m["date"], ci, ri))

    total_samples = len(sample_refs)
    print(f"  Total samples: {total_samples:,}")

    # Step 2: Sort by date and determine split boundaries
    sample_refs.sort(key=lambda x: x[0])
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    split_assignments = {"train": [], "val": [], "test": []}
    for i, (date, ci, ri) in enumerate(sample_refs):
        if i < train_end:
            split_assignments["train"].append((ci, ri))
        elif i < val_end:
            split_assignments["val"].append((ci, ri))
        else:
            split_assignments["test"].append((ci, ri))

    # Step 3: Write split .npy files by streaming chunks
    # Use .npy (raw) format with memory-mapped writing for zero extra RAM
    print("Writing split files to disk...")
    n_features = len(FEATURE_COLS)
    split_info = {}

    for name, refs in split_assignments.items():
        n = len(refs)
        if n == 0:
            continue

        x_path = os.path.join(chunk_dir, f"{name}_X.npy")
        y_path = os.path.join(chunk_dir, f"{name}_y.npy")

        # Create memory-mapped files
        X_mm = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float32,
            shape=(n, WINDOW_SIZE, n_features),
        )
        y_mm = np.lib.format.open_memmap(
            y_path, mode="w+", dtype=np.float32,
            shape=(n, 2),
        )

        # Group refs by chunk to minimize file reloads
        from collections import defaultdict
        chunk_to_rows = defaultdict(list)
        for out_idx, (ci, ri) in enumerate(refs):
            chunk_to_rows[ci].append((out_idx, ri))

        for ci, row_pairs in chunk_to_rows.items():
            data = np.load(chunk_files[ci][0])
            chunk_X, chunk_y = data["X"], data["y"]
            for out_idx, ri in row_pairs:
                X_mm[out_idx] = chunk_X[ri]
                y_mm[out_idx] = chunk_y[ri]
            del data, chunk_X, chunk_y

        X_mm.flush()
        y_mm.flush()
        del X_mm, y_mm

        # Collect meta for this split
        meta_list = []
        for ci, ri in refs:
            meta_list.append(sample_refs[0])  # placeholder — we need actual meta
        # Reload meta properly
        meta_list = []
        all_chunk_meta = {}
        for ci, ri in refs:
            if ci not in all_chunk_meta:
                with open(chunk_files[ci][1], "rb") as f:
                    all_chunk_meta[ci] = pickle.load(f)
            meta_list.append(all_chunk_meta[ci][ri])
        all_chunk_meta.clear()

        with open(os.path.join(chunk_dir, f"{name}_meta.pkl"), "wb") as f:
            pickle.dump(meta_list, f)

        split_info[name] = n
        date_range = f"{meta_list[0]['date']} to {meta_list[-1]['date']}"
        print(f"  {name}: {n:>8,} samples  ({date_range})")

    # Save scalers
    with open(os.path.join(chunk_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(all_scalers, f)

    # Return paths instead of arrays — model.py will use MemmapDataset
    splits = {}
    for name in ["train", "val", "test"]:
        x_path = os.path.join(chunk_dir, f"{name}_X.npy")
        y_path = os.path.join(chunk_dir, f"{name}_y.npy")
        meta_path = os.path.join(chunk_dir, f"{name}_meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        splits[name] = (x_path, y_path, meta)

    return splits, all_scalers


if __name__ == "__main__":
    splits, scalers = prepare_data()
    X_train, y_train, _ = splits["train"]
    print(f"\nReady for training:")
    print(f"  X_train: {X_train.shape}  (samples, window, features)")
    print(f"  y_train: {y_train.shape}  (close_return, dividend_yield)")
    print(f"  Close return — mean: {y_train[:, 0].mean():.6f}, std: {y_train[:, 0].std():.6f}")
    print(f"  Dividend yield — mean: {y_train[:, 1].mean():.6f}, std: {y_train[:, 1].std():.6f}")
