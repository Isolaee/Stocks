import yfinance as yf
import pandas as pd
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_top_100_by_market_cap():
    """Fetch S&P 500 tickers from Wikipedia, rank by market cap, return top 100."""
    print("Fetching S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    sp500 = tables[0]
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

    print(f"Fetching market caps for {len(tickers)} stocks (parallel)...")
    market_caps = {}

    def fetch_cap(symbol):
        try:
            info = yf.Ticker(symbol).info
            cap = info.get("marketCap")
            if cap:
                return symbol, cap
        except Exception:
            pass
        return symbol, None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch_cap, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            sym, cap = future.result()
            if cap:
                market_caps[sym] = cap
            if i % 50 == 0:
                print(f"  ...fetched {i}/{len(tickers)}")

    # Sort by market cap descending, take top 100
    ranked = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    top_100 = [sym for sym, _ in ranked[:100]]
    print(f"Top 100 stocks selected. Largest: {top_100[0]}, Smallest: {top_100[-1]}")
    return top_100


def build_features(symbol):
    """Build feature DataFrame for a single stock (5y history + fundamentals)."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5y")
    if hist.empty or len(hist) < 200:
        return None

    info = ticker.info
    df = hist.copy()

    # Derived price features
    df["Daily Return"] = df["Close"].pct_change()
    df["Volatility 20d"] = df["Daily Return"].rolling(20).std()
    df["MA 50"] = df["Close"].rolling(50).mean()
    df["MA 200"] = df["Close"].rolling(200).mean()
    df["MA 50/200 Ratio"] = df["MA 50"] / df["MA 200"]
    df["Volume MA 20"] = df["Volume"].rolling(20).mean()
    df["Volume Ratio"] = df["Volume"] / df["Volume MA 20"]
    df["High-Low Spread"] = df["High"] - df["Low"]
    df["Close-Open Spread"] = df["Close"] - df["Open"]
    df["Upper Shadow"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["Lower Shadow"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Momentum 10d"] = df["Close"] - df["Close"].shift(10)
    df["Momentum 30d"] = df["Close"] - df["Close"].shift(30)
    df["RSI 14"] = df["Daily Return"].rolling(14).apply(
        lambda x: (
            100 - 100 / (1 + x[x > 0].mean() / abs(x[x < 0].mean()))
            if abs(x[x < 0].mean()) > 0
            else 100
        ),
        raw=False,
    )
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    # Fundamental data as static columns
    fundamentals = {
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Profit Margin": info.get("profitMargins"),
        "Revenue Growth": info.get("revenueGrowth"),
        "Earnings Growth": info.get("earningsGrowth"),
        "Return on Equity": info.get("returnOnEquity"),
        "Debt to Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Free Cash Flow": info.get("freeCashflow"),
        "Operating Margin": info.get("operatingMargins"),
        "Beta": info.get("beta"),
        "Short Ratio": info.get("shortRatio"),
        "Payout Ratio": info.get("payoutRatio"),
        "Book Value": info.get("bookValue"),
        "Enterprise Value": info.get("enterpriseValue"),
        "EV/Revenue": info.get("enterpriseToRevenue"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
    }
    for name, value in fundamentals.items():
        if value is not None:
            df[name] = value

    df = df.dropna(axis=1, how="all")
    return df


def compute_correlations(df):
    """Compute Spearman correlations vs Close and vs Dividends for one stock."""
    numeric = df.select_dtypes(include="number")

    # Correlation with Close
    corr_close = numeric.corr(method="spearman")["Close"].drop("Close", errors="ignore")
    corr_close = corr_close.dropna()

    # Correlation with Dividends (only on dividend days)
    corr_div = None
    if "Dividends" in df.columns:
        div_days = df[df["Dividends"] > 0]
        if len(div_days) > 2:
            numeric_div = div_days.select_dtypes(include="number")
            corr_div = numeric_div.corr(method="spearman")["Dividends"].drop(
                "Dividends", errors="ignore"
            )
            corr_div = corr_div.dropna()

    return corr_close, corr_div


def print_ranked(title, corr_series):
    """Print a ranked correlation table with visual bars."""
    sorted_corr = corr_series.sort_values(ascending=False)
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)
    for i, (col, val) in enumerate(sorted_corr.items(), 1):
        bar = "+" * int(abs(val) * 20) if val > 0 else "-" * int(abs(val) * 20)
        print(f"{i:>3}. {col:<25} {val:>+.4f}  {bar}")


def main():
    # Step 1: Get top 100 stocks
    top_100 = get_top_100_by_market_cap()

    # Step 2: Collect correlations for each stock
    all_corr_close = []
    all_corr_div = []
    failed = []

    for i, symbol in enumerate(top_100, 1):
        print(f"[{i:>3}/100] {symbol}...", end=" ")
        try:
            df = build_features(symbol)
            if df is None:
                print("SKIP (insufficient data)")
                failed.append(symbol)
                continue

            corr_close, corr_div = compute_correlations(df)
            all_corr_close.append(corr_close)
            if corr_div is not None:
                all_corr_div.append(corr_div)
            print("OK")
        except Exception as e:
            print(f"FAIL ({e})")
            failed.append(symbol)

    # Step 3: Average correlations across all stocks
    print(f"\nProcessed {len(all_corr_close)} stocks, {len(failed)} failed: {failed}")

    avg_close = pd.DataFrame(all_corr_close).mean()
    avg_close = avg_close.dropna()

    print_ranked(
        f"AVERAGE CORRELATION WITH CLOSE PRICE (Spearman, {len(all_corr_close)} stocks)",
        avg_close,
    )

    if all_corr_div:
        avg_div = pd.DataFrame(all_corr_div).mean()
        avg_div = avg_div.dropna()
        print_ranked(
            f"AVERAGE CORRELATION WITH DIVIDENDS (Spearman, {len(all_corr_div)} stocks with dividends)",
            avg_div,
        )
    else:
        print("\nNo stocks had enough dividend data for correlation analysis.")
        avg_div = pd.Series(dtype=float)

    # Step 4: Save to CSV
    avg_close.sort_values(ascending=False).to_csv(
        "corr_close_avg.csv", header=["avg_spearman_corr"]
    )
    print("\nSaved: corr_close_avg.csv")

    if not avg_div.empty:
        avg_div.sort_values(ascending=False).to_csv(
            "corr_dividends_avg.csv", header=["avg_spearman_corr"]
        )
        print("Saved: corr_dividends_avg.csv")


if __name__ == "__main__":
    main()
