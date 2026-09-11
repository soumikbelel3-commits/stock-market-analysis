"""Technical indicators and statistical helpers.

Parameters default to the swing / positional horizon set in ``config``:
20/50/200 DMA, RSI(14), ATR(14), 63-session relative strength.

Wilder's smoothing is used for RSI and ATR (``ewm(alpha=1/n)``), which is what
charting packages plot -- a simple rolling mean gives visibly different values
and would make the scorecard disagree with the user's terminal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------
def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False, min_periods=max(2, window // 2)).mean()


def dma_set(close: pd.Series, windows=config.DMA_WINDOWS) -> pd.DataFrame:
    return pd.DataFrame({f"DMA{w}": sma(close, w) for w in windows})


# --------------------------------------------------------------------------
# Momentum / oscillators
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss.notna(), np.nan).fillna(
        pd.Series(100.0, index=close.index).where(avg_loss.eq(0) & avg_gain.gt(0)))


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


# --------------------------------------------------------------------------
# Volatility / range
# --------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    return pd.concat([(high - low).abs(),
                      (high - prev).abs(),
                      (low - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """Wilder's ATR in price units."""
    return true_range(df).ewm(alpha=1 / period, adjust=False,
                              min_periods=period).mean()


def atr_pct(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    return atr(df, period) / df["Close"] * 100


def realized_vol(close: pd.Series, window: int = config.VOL_WINDOW,
                 annualize: bool = True) -> pd.Series:
    r = close.pct_change()
    v = r.rolling(window, min_periods=max(3, window // 2)).std()
    return v * np.sqrt(TRADING_DAYS) * 100 if annualize else v


# --------------------------------------------------------------------------
# Drawdown / trend
# --------------------------------------------------------------------------
def drawdown(close: pd.Series) -> pd.Series:
    """Percentage below the running peak."""
    return (close / close.cummax() - 1) * 100


def max_drawdown(close: pd.Series) -> float:
    dd = drawdown(close)
    return float(dd.min()) if dd.notna().any() else np.nan


def cagr(close: pd.Series) -> float:
    s = close.dropna()
    if len(s) < 2:
        return np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0 or s.iloc[0] <= 0:
        return np.nan
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100


def trailing_return(close: pd.Series, days: int) -> float:
    s = close.dropna()
    if len(s) < 2:
        return np.nan
    cutoff = s.index[-1] - pd.Timedelta(days=days)
    past = s.loc[:cutoff]
    if past.empty:
        return np.nan
    return float(s.iloc[-1] / past.iloc[-1] - 1) * 100


def return_table(close: pd.Series) -> dict[str, float]:
    """Standard trailing-return row for a price series."""
    s = close.dropna()
    ytd = np.nan
    if len(s):
        jan1 = pd.Timestamp(year=s.index[-1].year, month=1, day=1)
        prior = s.loc[:jan1]
        if len(prior):
            ytd = float(s.iloc[-1] / prior.iloc[-1] - 1) * 100
    return {
        "1D_%": trailing_return(s, 1),
        "1W_%": trailing_return(s, 7),
        "1M_%": trailing_return(s, 30),
        "3M_%": trailing_return(s, 91),
        "6M_%": trailing_return(s, 182),
        "YTD_%": ytd,
        "1Y_%": trailing_return(s, 365),
        "3Y_%": trailing_return(s, 365 * 3),
        "5Y_%": trailing_return(s, 365 * 5),
        "CAGR_%": cagr(s),
        "MaxDD_%": max_drawdown(s),
        "Vol_%": float(realized_vol(s, 63).iloc[-1]) if len(s) > 63 else np.nan,
    }


# --------------------------------------------------------------------------
# Cross-sectional / relative
# --------------------------------------------------------------------------
def rel_strength(close: pd.Series, bench: pd.Series,
                 window: int = config.REL_STRENGTH_WINDOW) -> pd.Series:
    """Ratio line vs a benchmark, rebased to 100 at the start of the window."""
    joined = pd.concat([close, bench], axis=1, keys=["a", "b"]).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    ratio = joined["a"] / joined["b"]
    return ratio / ratio.iloc[0] * 100 if window is None else ratio / ratio.rolling(
        window, min_periods=2).mean() * 100


def rel_strength_score(close: pd.Series, bench: pd.Series,
                       window: int = config.REL_STRENGTH_WINDOW) -> float:
    """Excess return vs benchmark over `window` sessions, in percentage points."""
    j = pd.concat([close, bench], axis=1, keys=["a", "b"]).dropna()
    if len(j) < window + 1:
        window = max(2, len(j) - 1)
    if len(j) < 2:
        return np.nan
    a = j["a"].iloc[-1] / j["a"].iloc[-window - 1] - 1
    b = j["b"].iloc[-1] / j["b"].iloc[-window - 1] - 1
    return float((a - b) * 100)


def zscore(s: pd.Series | pd.DataFrame, window: int | None = None):
    """Z-score; rolling if a window is given, else full-sample."""
    if window:
        mu = s.rolling(window, min_periods=window // 2).mean()
        sd = s.rolling(window, min_periods=window // 2).std()
    else:
        mu, sd = s.mean(), s.std()
    return (s - mu) / sd


def percentile_rank(s: pd.Series, value: float | None = None) -> float:
    """Where the latest (or given) value sits in the series history, 0-100."""
    clean = s.dropna()
    if clean.empty:
        return np.nan
    v = clean.iloc[-1] if value is None else value
    return float((clean <= v).mean() * 100)


def rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    return s.rolling(window, min_periods=window // 4).apply(
        lambda w: (w <= w[-1]).mean() * 100, raw=True)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return j["a"].rolling(window, min_periods=window // 2).corr(j["b"])


# --------------------------------------------------------------------------
# Position / structure summaries used by the company scorecard
# --------------------------------------------------------------------------
def position_in_range(close: pd.Series, lookback: int = 252) -> float:
    """0 = at the period low, 100 = at the period high."""
    s = close.dropna().tail(lookback)
    if len(s) < 2:
        return np.nan
    lo, hi = s.min(), s.max()
    return np.nan if hi == lo else float((s.iloc[-1] - lo) / (hi - lo) * 100)


def dma_structure(close: pd.Series, windows=config.DMA_WINDOWS) -> dict:
    """Where price sits relative to each DMA, and whether the stack is bullish."""
    d = dma_set(close, windows)
    px = close.dropna().iloc[-1] if close.notna().any() else np.nan
    out = {"close": float(px)}
    vals = {}
    for w in windows:
        v = d[f"DMA{w}"].dropna()
        v = float(v.iloc[-1]) if len(v) else np.nan
        vals[w] = v
        out[f"px_vs_DMA{w}_%"] = np.nan if not np.isfinite(v) else float((px / v - 1) * 100)
    ordered = [vals[w] for w in sorted(windows)]
    out["stack_bullish"] = bool(
        all(np.isfinite(x) for x in ordered)
        and all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))
        and np.isfinite(px) and px >= ordered[0])
    out["above_all_dma"] = bool(
        np.isfinite(px) and all(np.isfinite(v) and px > v for v in vals.values()))
    out["n_dma_above"] = int(sum(1 for v in vals.values()
                                 if np.isfinite(v) and px > v))
    return out


def volume_trend(volume: pd.Series, short: int = 20, long: int = 63) -> float:
    """Short-run average volume vs long-run, as a percentage."""
    v = volume.dropna()
    if len(v) < long:
        return np.nan
    s, l = v.tail(short).mean(), v.tail(long).mean()
    return np.nan if l == 0 else float((s / l - 1) * 100)


def atr_levels(df: pd.DataFrame, period: int = config.ATR_PERIOD,
               lookback: int = 60) -> dict:
    """Structure-based level analysis derived from ATR and recent swing points.

    Not a recommendation -- it is the arithmetic a swing trader would do by
    hand: recent structure for the reference, ATR for the distances.
    """
    d = df.dropna(subset=["High", "Low", "Close"])
    if len(d) < max(period + 1, 20):
        return {}
    a = float(atr(d, period).iloc[-1])
    px = float(d["Close"].iloc[-1])
    recent = d.tail(lookback)
    swing_hi, swing_lo = float(recent["High"].max()), float(recent["Low"].min())
    return {
        "close": px,
        "ATR": a,
        "ATR_%": a / px * 100,
        "swing_high_60d": swing_hi,
        "swing_low_60d": swing_lo,
        "support_1atr": px - a,
        "support_2atr": px - 2 * a,
        "resistance_1atr": px + a,
        "resistance_2atr": px + 2 * a,
        "stop_2atr": px - 2 * a,
        "target_3atr": px + 3 * a,
        "reward_risk_at_3atr": 1.5,
        "dist_to_swing_high_%": (swing_hi / px - 1) * 100,
        "dist_to_swing_low_%": (swing_lo / px - 1) * 100,
        "dist_to_swing_high_atr": (swing_hi - px) / a if a else np.nan,
        "dist_to_swing_low_atr": (px - swing_lo) / a if a else np.nan,
    }


def technical_snapshot(df: pd.DataFrame, bench_close: pd.Series | None = None) -> dict:
    """Everything the technical leg of the scorecard needs, from one OHLCV frame."""
    close = df["Close"]
    snap = dma_structure(close)
    snap["RSI14"] = float(rsi(close).dropna().iloc[-1]) if rsi(close).notna().any() else np.nan
    snap["ATR14_%"] = float(atr_pct(df).dropna().iloc[-1]) if len(df) > config.ATR_PERIOD else np.nan
    snap["pos_52w_%"] = position_in_range(close, 252)
    snap["ret_1m_%"] = trailing_return(close, 30)
    snap["ret_3m_%"] = trailing_return(close, 91)
    snap["ret_6m_%"] = trailing_return(close, 182)
    snap["ret_12m_%"] = trailing_return(close, 365)
    snap["dd_from_high_%"] = float(drawdown(close).iloc[-1])
    snap["vol_ann_%"] = float(realized_vol(close, 63).dropna().iloc[-1]) if len(close) > 63 else np.nan
    if "Volume" in df.columns:
        snap["vol_trend_%"] = volume_trend(df["Volume"])
    if bench_close is not None:
        snap["rs_63d_pp"] = rel_strength_score(close, bench_close)
        snap["rs_126d_pp"] = rel_strength_score(close, bench_close, window=126)
    return snap


# --------------------------------------------------------------------------
# Traded value -- the input to every liquidity and cost calculation
# --------------------------------------------------------------------------
def adv_value(frames: dict[str, pd.DataFrame], window: int = 20,
              min_periods: int | None = None) -> pd.Series:
    """Average daily traded VALUE in rupees, per ticker. Not volume in shares.

    ``portfolio.liquidity_check``, ``costs.trade_cost`` and
    ``risk.liquidity_adjusted_var`` all consume this number and none of them
    computed it, which left every caller to roll its own -- and the obvious
    one-liner is wrong in two ways that both understate liquidity:

    * **A partially-formed last bar.** Yahoo serves the current session with a
      null close and a partial volume. Multiplying those gives NaN or a fraction
      of a day, and a plain ``.mean()`` over the window then averages a short day
      in as though it were a full one. Rows without both a close and a volume are
      dropped here before averaging.
    * **Averaging the product versus the product of averages.** The correct
      quantity is the mean of ``close * volume`` day by day. Taking mean price
      times mean volume is a different number whenever price and volume covary,
      which on a stock is always.

    Returns rupees per day, which is what the consumers document that they want.
    """
    mp = max(3, window // 2) if min_periods is None else int(min_periods)
    out = {}
    for t, d in frames.items():
        if not isinstance(d, pd.DataFrame) or not {"Close", "Volume"} <= set(d.columns):
            continue
        j = pd.DataFrame({"c": pd.to_numeric(d["Close"], errors="coerce"),
                          "v": pd.to_numeric(d["Volume"], errors="coerce")}).dropna()
        j = j[j["v"] > 0]
        if len(j) < mp:
            out[t] = np.nan
            continue
        out[t] = float((j["c"] * j["v"]).tail(window).mean())
    ser = pd.Series(out, name="adv_value", dtype="float64")
    ser.attrs["window"] = window
    ser.attrs["units"] = "INR per day"
    ser.attrs["n_missing"] = int(ser.isna().sum())
    return ser


def daily_vol(frames_or_returns, window: int = 63) -> pd.Series:
    """Daily return standard deviation per name, as a FRACTION.

    The companion to ``adv_value``: the impact model and the SPAN proxy both take
    a daily sigma as a fraction, while ``realized_vol`` returns an annualised
    percentage. Converting between the two by hand at each call site is exactly
    how a factor of 15.87 ends up in a margin number.
    """
    if isinstance(frames_or_returns, pd.DataFrame):
        rets = frames_or_returns
    else:
        cols = {t: pd.to_numeric(d["Close"], errors="coerce").pct_change()
                for t, d in frames_or_returns.items()
                if isinstance(d, pd.DataFrame) and "Close" in d.columns}
        rets = pd.DataFrame(cols)
    out = rets.tail(window).std(ddof=1)
    out.name = "daily_vol"
    out.attrs["window"] = window
    out.attrs["units"] = "fraction of price per day"
    return out
