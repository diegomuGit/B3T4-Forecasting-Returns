"""
data_loader.py — Download price data and prepare inputs for the backtest.

Responsibilities
----------------
1. Download adjusted close prices via yfinance for the full lookback + backtest period.
2. Compute log-prices and the frac-diff series.
3. At each rebalancing date, build the model input window (V_IN days of scaled frac-diff).

Strategy: downloads each ticker individually to avoid yfinance 1.2+ MultiIndex
issues that cause empty DataFrames when downloading many tickers at once.
"""

import numpy as np
import pandas as pd
import time


def _download_single_ticker(ticker: str,
                            start: str,
                            end: str,
                            max_retries: int = 3) -> pd.Series:
    """
    Download adjusted close for a single ticker.  Returns a named pd.Series.
    Retries with exponential backoff on failure.
    """
    import yfinance as yf

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                # Single ticker: columns are price types (Close, High, ...)
                # or MultiIndex ('Close', ticker) in yfinance >= 1.2
                if isinstance(df.columns, pd.MultiIndex):
                    series = df['Close'][ticker]
                elif 'Close' in df.columns:
                    series = df['Close']
                else:
                    series = df.iloc[:, 0]
                series.name = ticker
                return series
        except Exception:
            pass

        if attempt < max_retries:
            time.sleep(1.5 ** attempt)

    raise RuntimeError(f'Failed to download {ticker} after {max_retries} attempts.')


def download_prices(tickers: list,
                    start: str,
                    end: str,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance, one ticker at a time.

    Returns
    -------
    df : pd.DataFrame, shape (T, N_ASSETS)
        DatetimeIndex, columns = tickers.  No NaNs (forward-filled, then
        back-filled for the very first rows if needed).
    """
    # Clean date strings (remove time component from pd.DateOffset results)
    start = str(pd.Timestamp(start).date())
    end   = str(pd.Timestamp(end).date())

    if verbose:
        import yfinance as yf
        print(f'> yfinance version : {yf.__version__}')
        print(f'> Downloading {len(tickers)} tickers individually ({start} -> {end}) ...')

    series_list = []
    failed = []
    for i, ticker in enumerate(tickers):
        try:
            s = _download_single_ticker(ticker, start, end)
            series_list.append(s)
            if verbose:
                print(f'  [{i+1:2d}/{len(tickers)}] {ticker:>5s}  {len(s):,} days', end='')
                print(f'  [{s.index[0].date()} -> {s.index[-1].date()}]')
        except RuntimeError as e:
            failed.append(ticker)
            if verbose:
                print(f'  [{i+1:2d}/{len(tickers)}] {ticker:>5s}  FAILED')

    if failed:
        raise RuntimeError(
            f'Failed to download: {failed}\n'
            f'Check your internet connection and that these tickers are valid.'
        )

    # Combine into a single DataFrame
    df = pd.concat(series_list, axis=1)

    # Handle NaN values (holidays / slight date mismatches across tickers)
    n_nan_before = df.isna().sum().sum()
    df = df.ffill().bfill()
    n_nan_after = df.isna().sum().sum()

    if verbose:
        print(f'\n> Combined: {df.shape[0]:,} trading days x {df.shape[1]} assets')
        print(f'> Range: {df.index[0].date()} -> {df.index[-1].date()}')
        if n_nan_before > 0:
            print(f'  {n_nan_before} NaN values found; {n_nan_before - n_nan_after} filled.')
        if n_nan_after > 0:
            print(f'  WARNING: {n_nan_after} NaN values remain after fill!')
        else:
            print(f'  No NaN values.')

    return df


def get_trading_dates(prices_df: pd.DataFrame,
                      bt_start: str,
                      bt_end: str) -> pd.DatetimeIndex:
    """
    Return the subset of the price index that falls within [bt_start, bt_end].
    These are actual trading dates (the market was open).
    """
    mask = (prices_df.index >= pd.Timestamp(bt_start)) & \
           (prices_df.index <= pd.Timestamp(bt_end))
    return prices_df.index[mask]


def compute_rebalancing_dates(trading_dates: pd.DatetimeIndex,
                              rebal_days: int) -> list:
    """
    Build the list of rebalancing dates: the first trading day in the period,
    then every *rebal_days* trading days after that.

    Returns a list of pd.Timestamp.
    """
    dates = [trading_dates[0]]
    idx = 0
    while idx + rebal_days < len(trading_dates):
        idx += rebal_days
        dates.append(trading_dates[idx])
    return dates


def prepare_log_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log(price) for the entire dataframe.
    """
    return np.log(prices_df)
