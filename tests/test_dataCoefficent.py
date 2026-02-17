import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from dataCoefficent import build_features, compute_correlations, print_ranked


def make_mock_history(days=300):
    """Create realistic mock stock history DataFrame."""
    dates = pd.bdate_range(end="2025-01-01", periods=days)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    return pd.DataFrame(
        {
            "Open": close - np.random.rand(days),
            "High": close + np.abs(np.random.randn(days)),
            "Low": close - np.abs(np.random.randn(days)),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, days),
            "Dividends": np.where(np.arange(days) % 63 == 0, 0.82, 0),
            "Stock Splits": np.zeros(days),
        },
        index=dates,
    )


MOCK_INFO = {
    "marketCap": 3_000_000_000_000,
    "trailingPE": 30.5,
    "forwardPE": 28.0,
    "pegRatio": 1.5,
    "priceToBook": 40.0,
    "dividendYield": 0.005,
    "profitMargins": 0.25,
    "revenueGrowth": 0.08,
    "earningsGrowth": 0.12,
    "returnOnEquity": 1.5,
    "debtToEquity": 150.0,
    "currentRatio": 1.1,
    "freeCashflow": 100_000_000_000,
    "operatingMargins": 0.30,
    "beta": 1.2,
    "shortRatio": 1.5,
    "payoutRatio": 0.15,
    "bookValue": 4.0,
    "enterpriseValue": 3_100_000_000_000,
    "enterpriseToRevenue": 8.0,
    "enterpriseToEbitda": 25.0,
}


class TestBuildFeatures:
    @patch("dataCoefficent.yf.Ticker")
    def test_returns_dataframe_with_derived_columns(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_mock_history(300)
        mock_ticker.info = MOCK_INFO
        mock_ticker_cls.return_value = mock_ticker

        df = build_features("AAPL")

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 300
        # Check derived features exist
        for col in [
            "Daily Return", "Volatility 20d", "MA 50", "MA 200",
            "MA 50/200 Ratio", "RSI 14", "Cumulative Return",
            "Momentum 10d", "Momentum 30d", "High-Low Spread",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    @patch("dataCoefficent.yf.Ticker")
    def test_returns_none_for_insufficient_data(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_mock_history(50)
        mock_ticker_cls.return_value = mock_ticker

        result = build_features("TINY")
        assert result is None

    @patch("dataCoefficent.yf.Ticker")
    def test_returns_none_for_empty_history(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = build_features("EMPTY")
        assert result is None

    @patch("dataCoefficent.yf.Ticker")
    def test_fundamentals_added_as_columns(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_mock_history(300)
        mock_ticker.info = MOCK_INFO
        mock_ticker_cls.return_value = mock_ticker

        df = build_features("AAPL")
        for col in ["Market Cap", "PE Ratio", "Beta", "Dividend Yield"]:
            assert col in df.columns
            assert (df[col] == MOCK_INFO[{
                "Market Cap": "marketCap",
                "PE Ratio": "trailingPE",
                "Beta": "beta",
                "Dividend Yield": "dividendYield",
            }[col]]).all()

    @patch("dataCoefficent.yf.Ticker")
    def test_missing_fundamentals_skipped(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_mock_history(300)
        mock_ticker.info = {"marketCap": 1000}  # most fields missing
        mock_ticker_cls.return_value = mock_ticker

        df = build_features("SPARSE")
        assert df is not None
        assert "Market Cap" in df.columns
        assert "PE Ratio" not in df.columns


class TestComputeCorrelations:
    def test_returns_close_correlations(self):
        df = make_mock_history(300)
        df["MA 50"] = df["Close"].rolling(50).mean()
        corr_close, corr_div = compute_correlations(df)

        assert isinstance(corr_close, pd.Series)
        assert "Close" not in corr_close.index
        assert len(corr_close) > 0
        # All correlation values should be between -1 and 1
        assert (corr_close.abs() <= 1.0).all()

    def test_returns_dividend_correlations_when_enough_data(self):
        df = make_mock_history(300)
        corr_close, corr_div = compute_correlations(df)

        # Mock data has dividends every 63 days -> ~4-5 dividend days
        assert corr_div is not None
        assert isinstance(corr_div, pd.Series)
        assert "Dividends" not in corr_div.index

    def test_returns_none_for_dividends_when_insufficient(self):
        df = make_mock_history(300)
        df["Dividends"] = 0  # no dividend days
        corr_close, corr_div = compute_correlations(df)

        assert corr_div is None

    def test_close_correlation_with_open_is_high(self):
        df = make_mock_history(300)
        corr_close, _ = compute_correlations(df)

        assert "Open" in corr_close.index
        assert corr_close["Open"] > 0.9


class TestPrintRanked:
    def test_prints_output(self, capsys):
        series = pd.Series({"MA 50": 0.95, "Volume": -0.20, "RSI": 0.05})
        print_ranked("TEST TITLE", series)

        captured = capsys.readouterr()
        assert "TEST TITLE" in captured.out
        assert "MA 50" in captured.out
        assert "Volume" in captured.out
        assert "+0.9500" in captured.out
        assert "-0.2000" in captured.out

    def test_sorts_descending(self, capsys):
        series = pd.Series({"A": 0.1, "B": 0.9, "C": -0.5})
        print_ranked("SORT TEST", series)

        captured = capsys.readouterr()
        lines = [l for l in captured.out.split("\n") if ". " in l]
        assert "B" in lines[0]
        assert "A" in lines[1]
        assert "C" in lines[2]
