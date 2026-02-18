import yfinance as yf
import pandas as pd
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def progress_bar(current, total, label="", width=40):
    """Print a simple progress bar to stderr."""
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  {bar} {current}/{total} ({pct:.0%}) {label}")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\n")


def get_sp500_tickers():
    """Fetch all S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    sp500 = tables[0]
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"Found {len(tickers)} S&P 500 tickers")
    return tickers


def fetch_all_history(tickers, start="2000-01-01", end="2025-12-31"):
    """Download daily OHLCV history for each ticker. Returns a single DataFrame."""
    all_frames = []
    total = len(tickers)
    failed = []

    print(f"Downloading history for {total} stocks ({start} to {end})...")

    for i, symbol in enumerate(tickers, 1):
        progress_bar(i, total, label=symbol)
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end, auto_adjust=False)
            if hist.empty:
                failed.append(symbol)
                continue
            hist["Symbol"] = symbol
            all_frames.append(hist)
        except Exception:
            failed.append(symbol)

    print(f"\nDownloaded {len(all_frames)} stocks, {len(failed)} failed")
    if failed:
        print(f"Failed tickers: {failed}")

    df = pd.concat(all_frames)
    df.index.name = "Date"
    return df


MACRO_TICKERS = {
    "^VIX": "VIX",
    "^GSPC": "SP500",
    "^TNX": "Treasury_10Y",
    "DX-Y.NYB": "USD_Index",
}


def fetch_macro_history(start="2000-01-01", end="2025-12-31"):
    """Download macro indicator history (VIX, S&P 500, 10Y Treasury, USD Index).

    Returns a DataFrame with Date index and columns:
    VIX, SP500_Return, Treasury_10Y, USD_Index
    """
    print(f"\nDownloading macro indicators ({start} to {end})...")
    frames = {}

    for ticker_symbol, name in MACRO_TICKERS.items():
        print(f"  Fetching {name} ({ticker_symbol})...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start, end=end, auto_adjust=False)
            if hist.empty:
                print(f"  WARNING: No data for {ticker_symbol}")
                continue
            frames[name] = hist["Close"]
        except Exception as e:
            print(f"  ERROR fetching {ticker_symbol}: {e}")

    macro = pd.DataFrame(frames)
    macro.index.name = "Date"

    # S&P 500: convert price to daily return (more useful than raw level)
    if "SP500" in macro.columns:
        macro["SP500_Return"] = macro["SP500"].pct_change()
        macro = macro.drop(columns=["SP500"])

    # Forward-fill weekday gaps (some indices have different trading calendars)
    macro = macro.ffill()

    print(f"  Macro data: {len(macro):,} rows, columns: {list(macro.columns)}")
    return macro


def main():
    tickers = get_sp500_tickers()

    df = fetch_all_history(tickers, start="2000-01-01", end="2025-12-31")

    out_path = "sp500_history.csv"
    print(f"Saving to {out_path}...")
    df.to_csv(out_path)

    rows = len(df)
    symbols = df["Symbol"].nunique()
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Done: {rows:,} rows, {symbols} stocks, ~{size_mb:.0f} MB in memory")

    # Macro indicators
    macro = fetch_macro_history(start="2000-01-01", end="2025-12-31")
    macro_path = "macro_history.csv"
    print(f"Saving macro data to {macro_path}...")
    macro.to_csv(macro_path)
    print(f"Macro done: {len(macro):,} rows")


if __name__ == "__main__":
    main()
