import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty 200 Order Flow Scanner", page_icon="📊", layout="wide")

st.title("📊 Nifty 200 — Order Flow + Delta Scanner")
st.caption("Multi-timeframe technical structure + volume + Delta/CVD confirmation. Real footprint data can be supplied by CSV; otherwise the app clearly labels an OHLCV Delta proxy.")

NIFTY200_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
BUY_THRESHOLD = 70
SELL_THRESHOLD = 70
STRONG_THRESHOLD = 82

@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty200():
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/"}
    try:
        r = requests.get(NIFTY200_URL, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig", errors="ignore")))
        col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
        if col is None:
            raise ValueError("Symbol column not found")
        symbols = df[col].astype(str).str.strip().str.upper().replace("", np.nan).dropna().drop_duplicates().tolist()
        if len(symbols) < 150:
            raise ValueError(f"Only {len(symbols)} symbols returned")
        return symbols, df
    except Exception:
        fallback = "https://raw.githubusercontent.com/anirbanghoshsbi/data/master/ind_nifty200list.csv"
        r = requests.get(fallback, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
        if col is None:
            raise ValueError("Fallback symbol column not found")
        symbols = df[col].astype(str).str.strip().str.upper().replace("", np.nan).dropna().drop_duplicates().tolist()
        return symbols, df

def clean_ohlcv(data):
    if data is None or data.empty:
        return pd.DataFrame()
    x = data.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in x.columns for c in needed):
        return pd.DataFrame()
    x = x[needed].dropna(subset=["Open", "High", "Low", "Close"])
    x["Volume"] = pd.to_numeric(x["Volume"], errors="coerce").fillna(0)
    return x[~x.index.duplicated(keep="last")].sort_index()

def resample_ohlcv(df, rule):
    if df.empty:
        return df
    return df.resample(rule, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s):
    line = ema(s, 12) - ema(s, 26)
    signal = ema(line, 9)
    return line, signal, line - signal

def session_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    dates = pd.Series(df.index.date, index=df.index)
    return pv.groupby(dates).cumsum() / df["Volume"].groupby(dates).cumsum().replace(0, np.nan)

def session_volume_profile(df, value_area_pct=0.70, bins=48):
    """Approximate session POC/VAH/VAL from OHLCV bars.
    This is NOT footprint price-level volume; genuine price-level profile data
    must come from a footprint/volume-profile export.
    """
    if df.empty:
        return np.nan, np.nan, np.nan
    x = df.tail(min(len(df), 240)).copy()
    lo = float(x["Low"].min())
    hi = float(x["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins, dtype=float)
    for _, row in x.iterrows():
        low = float(row["Low"]); high = float(row["High"]); v = float(row["Volume"])
        if not np.isfinite(v) or v <= 0:
            continue
        if high <= low:
            idx = int(np.clip(np.searchsorted(edges, float(row["Close"])) - 1, 0, bins - 1))
            vol[idx] += v
            continue
        touched = np.where((mids >= low) & (mids <= high))[0]
        if len(touched) == 0:
            idx = int(np.clip(np.searchsorted(edges, (low + high) / 2) - 1, 0, bins - 1))
            vol[idx] += v
        else:
            vol[touched] += v / len(touched)
    if vol.sum() <= 0:
        return np.nan, np.nan, np.nan
    poc_i = int(np.argmax(vol))
    target = vol.sum() * value_area_pct
    selected = {poc_i}
    total = vol[poc_i]
    left = poc_i - 1
    right = poc_i + 1
    while total < target and (left >= 0 or right < bins):
        lv = vol[left] if left >= 0 else -1
        rv = vol[right] if right < bins else -1
        if rv > lv:
            selected.add(right); total += rv; right += 1
        elif left >= 0:
            selected.add(left); total += lv; left -= 1
        else:
            break
    va_low = float(mids[min(selected)])
    va_high = float(mids[max(selected)])
    return float(mids[poc_i]), va_high, va_low

def add_indicators(df):
    x = df.copy()
    x["EMA9"] = ema(x["Close"], 9)
    x["EMA20"] = ema(x["Close"], 20)
    x["EMA50"] = ema(x["Close"], 50)
    x["EMA200"] = ema(x["Close"], 200)
    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["VWAP"] = session_vwap(x)
    x["MACD"], x["MACD_SIGNAL"], x["MACD_HIST"] = macd(x["Close"])
    x["VOL_AVG20"] = sma(x["Volume"], 20)
    x["VOL_RATIO"] = x["Volume"] / x["VOL_AVG20"].replace(0, np.nan)
    x["HH20"] = x["High"].rolling(20).max().shift(1)
    x["LL20"] = x["Low"].rolling(20).min().shift(1)

    # IMPORTANT: this is a pressure proxy, NOT bid/ask footprint Delta.
    rng = (x["High"] - x["Low"]).replace(0, np.nan)
    clv = (((x["Close"] - x["Low"]) - (x["High"] - x["Close"])) / rng).clip(-1, 1)
    x["DELTA_PROXY"] = clv * x["Volume"]
    x["DELTA_PROXY_PCT"] = x["DELTA_PROXY"] / x["Volume"].replace(0, np.nan) * 100
    x["CVD_PROXY"] = x["DELTA_PROXY"].cumsum()
    x["CVD_SLOPE"] = x["CVD_PROXY"] - x["CVD_PROXY"].shift(20)
    x["BODY"] = x["Close"] - x["Open"]
    x["BODY_RANGE"] = x["BODY"] / rng
    return x

@st.cache_data(show_spinner=False)
def read_orderflow(uploaded_items):
    """Read GoCharting chart-data CSV exports with flexible column names."""
    if not uploaded_items:
        return pd.DataFrame()
    if not isinstance(uploaded_items, list):
        uploaded_items = [uploaded_items]
    frames = []
    aliases = {
        "symbol": ["symbol", "ticker", "instrument", "security"],
        "datetime": ["datetime", "date time", "date_time", "timestamp", "time", "date"],
        "delta": ["delta", "orderflow delta", "bar delta", "of delta"],
        "buyvolume": ["buyvolume", "buy volume", "buy_volume", "ask volume", "askvolume", "aggressive buy volume"],
        "sellvolume": ["sellvolume", "sell volume", "sell_volume", "bid volume", "bidvolume", "aggressive sell volume"],
        "maxdelta": ["maxdelta", "max delta", "max_delta"],
        "mindelta": ["mindelta", "min delta", "min_delta"],
        "cvd": ["cvd", "cumulative delta", "cumulative volume delta"],
        "buytrades": ["buy", "buy trades", "buytrades"],
        "selltrades": ["sell", "sell trades", "selltrades"],
        "trades": ["trades", "total trades", "trade count"],
    }
    def norm(s):
        return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())
    def find_col(columns, names):
        lookup = {norm(col): col for col in columns}
        for name in names:
            if norm(name) in lookup:
                return lookup[norm(name)]
        return None
    for item in uploaded_items:
        try:
            raw = item.getvalue() if hasattr(item, "getvalue") else item
            x = pd.read_csv(io.BytesIO(raw))
            mapping = {}
            for target, names in aliases.items():
                col = find_col(x.columns, names)
                if col is not None:
                    mapping[target] = col
            if "datetime" not in mapping or "delta" not in mapping:
                continue
            out = pd.DataFrame()
            out["DateTime"] = pd.to_datetime(x[mapping["datetime"]], errors="coerce")
            if "symbol" in mapping:
                out["Symbol"] = x[mapping["symbol"]].astype(str).str.upper().str.strip()
            else:
                filename = getattr(item, "name", "")
                inferred = str(filename).upper().replace(".CSV", "").replace(".NS", "")
                out["Symbol"] = inferred.strip()
            for target in ["delta", "buyvolume", "sellvolume", "maxdelta", "mindelta", "cvd", "buytrades", "selltrades", "trades"]:
                out[target] = pd.to_numeric(x[mapping[target]], errors="coerce") if target in mapping else np.nan
            out["Symbol"] = out["Symbol"].str.replace(r"[^A-Z0-9&_-]", "", regex=True)
            out["Symbol"] = out["Symbol"].str.replace("_NS", "", regex=False).str.replace("-NS", "", regex=False)
            out = out.dropna(subset=["DateTime", "Symbol", "delta"])
            if not out.empty:
                frames.append(out)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["Symbol", "DateTime"], keep="last")
    return result.sort_values(["Symbol", "DateTime"]).reset_index(drop=True)

def orderflow_analysis(symbol, df, real_of):
    """Score genuine footprint/order-flow data, with safe OHLCV fallback."""
    r = df.iloc[-1]
    if not real_of.empty:
        rows = real_of[real_of["Symbol"] == symbol].sort_values("DateTime").copy()
        if not rows.empty:
            row = rows.iloc[-1]
            delta = pd.to_numeric(row.get("delta"), errors="coerce")
            buy = pd.to_numeric(row.get("buyvolume"), errors="coerce")
            sell = pd.to_numeric(row.get("sellvolume"), errors="coerce")
            max_delta = pd.to_numeric(row.get("maxdelta"), errors="coerce")
            min_delta = pd.to_numeric(row.get("mindelta"), errors="coerce")
            cvd = pd.to_numeric(row.get("cvd"), errors="coerce")

            score = 0
            reasons = []

            # Delta direction and relative strength.
            side_total = np.nan
            delta_pct = np.nan
            if pd.notna(buy) and pd.notna(sell):
                side_total = buy + sell
                if side_total > 0:
                    delta_pct = (buy - sell) / side_total * 100

            if pd.notna(delta):
                if pd.notna(side_total) and side_total > 0:
                    ratio = abs(delta) / side_total
                else:
                    ratio = np.nan
                if delta > 0:
                    score += 3
                    reasons.append("positive footprint Delta")
                    if pd.notna(ratio) and ratio >= 0.25:
                        score += 2
                        reasons.append("strong positive Delta/volume")
                elif delta < 0:
                    score -= 3
                    reasons.append("negative footprint Delta")
                    if pd.notna(ratio) and ratio >= 0.25:
                        score -= 2
                        reasons.append("strong negative Delta/volume")

            # Buy/sell volume dominance.
            if pd.notna(delta_pct):
                if delta_pct >= 25:
                    score += 2
                    reasons.append("buy volume dominates")
                elif delta_pct <= -25:
                    score -= 2
                    reasons.append("sell volume dominates")

            # Intrabar Delta extremes: useful for absorption/exhaustion context.
            if pd.notna(max_delta) and pd.notna(min_delta):
                if max_delta > 0 and min_delta < 0:
                    swing = max_delta + abs(min_delta)
                    if pd.notna(delta) and swing > 0 and abs(delta) <= 0.20 * swing:
                        reasons.append("large two-sided Delta swing with weak close")
                        if delta >= 0:
                            score -= 1
                        else:
                            score += 1

                if pd.notna(delta):
                    if delta > 0 and max_delta > 0 and abs(min_delta) >= 1.5 * abs(max_delta):
                        score -= 2
                        reasons.append("buying pressure absorbed by strong negative excursion")
                    elif delta < 0 and min_delta < 0 and max_delta >= 1.5 * abs(min_delta):
                        score += 2
                        reasons.append("selling pressure absorbed by strong positive excursion")

            # CVD direction and acceleration.
            cvd_change_5 = np.nan
            cvd_change_10 = np.nan
            if "cvd" in rows.columns and len(rows) >= 5:
                cvd_series = pd.to_numeric(rows["cvd"], errors="coerce")
                cvd_now = cvd_series.iloc[-1]
                cvd_old5 = cvd_series.iloc[-5]
                if pd.notna(cvd_now) and pd.notna(cvd_old5):
                    cvd_change_5 = cvd_now - cvd_old5
                    if cvd_change_5 > 0:
                        score += 2
                        reasons.append("CVD rising")
                    elif cvd_change_5 < 0:
                        score -= 2
                        reasons.append("CVD falling")
            if "cvd" in rows.columns and len(rows) >= 10:
                cvd_series = pd.to_numeric(rows["cvd"], errors="coerce")
                a = cvd_series.iloc[-5]
                b = cvd_series.iloc[-10]
                if pd.notna(a) and pd.notna(b):
                    cvd_change_10 = a - b
                    if pd.notna(cvd_change_5):
                        if cvd_change_5 > cvd_change_10 > 0:
                            score += 1
                            reasons.append("CVD acceleration positive")
                        elif cvd_change_5 < cvd_change_10 < 0:
                            score -= 1
                            reasons.append("CVD acceleration negative")

            # Trade-count confirmation, if supplied.
            buy_trades = pd.to_numeric(row.get("buytrades"), errors="coerce")
            sell_trades = pd.to_numeric(row.get("selltrades"), errors="coerce")
            trades = pd.to_numeric(row.get("trades"), errors="coerce")
            if pd.notna(buy_trades) and pd.notna(sell_trades) and buy_trades + sell_trades > 0:
                trade_delta_pct = (buy_trades - sell_trades) / (buy_trades + sell_trades) * 100
                if trade_delta_pct >= 30:
                    score += 1
                    reasons.append("buy-trade count dominates")
                elif trade_delta_pct <= -30:
                    score -= 1
                    reasons.append("sell-trade count dominates")

            # Normalized extreme-Deltas for cross-stock comparison.
            extreme_ratio = np.nan
            if pd.notna(max_delta) and pd.notna(min_delta) and pd.notna(side_total) and side_total > 0:
                extreme_ratio = (max_delta - min_delta) / side_total
                if extreme_ratio >= 1.0:
                    reasons.append("wide intrabar Delta range")

            direction = "BUY" if score >= 4 else "SELL" if score <= -4 else "NEUTRAL"
            return {
                "source": "GOCHARTING_REAL",
                "delta": delta, "buy": buy, "sell": sell,
                "max_delta": max_delta, "min_delta": min_delta, "cvd": cvd,
                "delta_pct": delta_pct, "cvd_change_5": cvd_change_5,
                "extreme_ratio": extreme_ratio, "trades": trades,
                "score": score, "direction": direction,
                "reason": "; ".join(reasons)
            }

    # Fallback: OHLCV pressure proxy, clearly marked as non-footprint data.
    delta_pct = r["DELTA_PROXY_PCT"]
    cvd_slope = r["CVD_SLOPE"]
    score = 0
    reasons = []
    if pd.notna(delta_pct):
        if delta_pct >= 35:
            score += 3; reasons.append("strong positive Delta proxy")
        elif delta_pct >= 15:
            score += 2; reasons.append("positive Delta proxy")
        elif delta_pct <= -35:
            score -= 3; reasons.append("strong negative Delta proxy")
        elif delta_pct <= -15:
            score -= 2; reasons.append("negative Delta proxy")
    if pd.notna(cvd_slope):
        if cvd_slope > 0:
            score += 2; reasons.append("rising CVD proxy")
        elif cvd_slope < 0:
            score -= 2; reasons.append("falling CVD proxy")
    if pd.notna(delta_pct) and pd.notna(r["BODY_RANGE"]) and abs(delta_pct) >= 45 and abs(r["BODY_RANGE"]) <= 0.25:
        if delta_pct > 0:
            score -= 1; reasons.append("possible buying absorption proxy")
        else:
            score += 1; reasons.append("possible selling absorption proxy")
    return {
        "source": "OHLCV_PROXY", "delta": np.nan, "buy": np.nan, "sell": np.nan,
        "max_delta": np.nan, "min_delta": np.nan, "cvd": r["CVD_PROXY"],
        "delta_pct": delta_pct, "cvd_change_5": np.nan, "extreme_ratio": np.nan,
        "trades": np.nan, "score": score,
        "direction": "BUY" if score >= 3 else "SELL" if score <= -3 else "NEUTRAL",
        "reason": "; ".join(reasons)
    }

def timeframe_score(df, label):
    if df.empty or len(df) < 30:
        return 0, f"{label}: insufficient data"
    r = df.iloc[-1]
    score = 0
    reasons = []
    for col, points in [("EMA20", 2), ("EMA50", 2), ("EMA200", 2)]:
        if pd.notna(r[col]):
            if r["Close"] > r[col]:
                score += points; reasons.append(f"{label} above {col}")
            else:
                score -= points; reasons.append(f"{label} below {col}")
    if pd.notna(r["MACD_HIST"]):
        score += 1 if r["MACD_HIST"] > 0 else -1
    return score, "; ".join(reasons)

def scan_symbol(symbol, period, interval, real_of, downloaded=None):
    try:
        # Market data is downloaded in batches before scanning. This avoids
        # hundreds of simultaneous Yahoo requests, which can trigger
        # throttling/empty responses on Streamlit Cloud.
        if downloaded is not None and symbol in downloaded:
            data = downloaded[symbol]
        else:
            data = yf.download(
                symbol + ".NS",
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )
        df5 = clean_ohlcv(data)
        if len(df5) < 80:
            raise ValueError("insufficient intraday data")
        df5 = add_indicators(df5)
        df15 = add_indicators(resample_ohlcv(df5, "15min"))
        df60 = add_indicators(resample_ohlcv(df5, "60min"))
        r = df5.iloc[-1]
        poc, vah, val = session_volume_profile(df5)

        score = 0
        reasons = []
        for d, label in [(df5, "5m"), (df15, "15m"), (df60, "60m")]:
            s, rs = timeframe_score(d, label)
            score += s * 2
            if rs:
                reasons.append(rs)

        if pd.notna(r["VWAP"]):
            if r["Close"] > r["VWAP"]:
                score += 5; reasons.append("above VWAP")
            else:
                score -= 5; reasons.append("below VWAP")

        if pd.notna(r["VOL_RATIO"]):
            if r["VOL_RATIO"] >= 2:
                score += 5 if r["BODY"] > 0 else -5
                reasons.append(f"volume {r['VOL_RATIO']:.1f}x average")
            elif r["VOL_RATIO"] >= 1.3:
                score += 3 if r["BODY"] > 0 else -3

        if pd.notna(r["HH20"]) and r["Close"] > r["HH20"]:
            score += 7; reasons.append("20-bar breakout")
        elif pd.notna(r["LL20"]) and r["Close"] < r["LL20"]:
            score -= 7; reasons.append("20-bar breakdown")

        if pd.notna(r["RSI"]):
            if 55 <= r["RSI"] <= 72:
                score += 3; reasons.append("bullish RSI")
            elif 28 <= r["RSI"] <= 45:
                score -= 3; reasons.append("bearish RSI")

        if pd.notna(r["MACD_HIST"]):
            score += 3 if r["MACD_HIST"] > 0 else -3

        # Approximate volume-profile context from OHLCV.
        if pd.notna(poc) and pd.notna(vah) and pd.notna(val):
            if r["Close"] > vah:
                score += 3
                reasons.append("price above approximate VAH")
            elif r["Close"] < val:
                score -= 3
                reasons.append("price below approximate VAL")
            elif r["Close"] >= poc:
                score += 1
                reasons.append("price at/above approximate POC")
            else:
                score -= 1
                reasons.append("price below approximate POC")

        of = orderflow_analysis(symbol, df5, real_of)
        score += of["score"] * (5 if of["source"] == "GOCHARTING_REAL" else 3)
        if of["reason"]:
            reasons.append(of["reason"])

        if of["source"] == "GOCHARTING_REAL":
            if pd.notna(of["buy"]) and pd.notna(of["sell"]):
                total_side_volume = float(of["buy"]) + float(of["sell"])
                if total_side_volume > 0:
                    buy_share = float(of["buy"]) / total_side_volume
                    if buy_share >= 0.65:
                        score += 3; reasons.append("buy volume dominates sell volume")
                    elif buy_share <= 0.35:
                        score -= 3; reasons.append("sell volume dominates buy volume")
            prior = real_of[real_of["Symbol"] == symbol].sort_values("DateTime")
            if "cvd" in prior.columns and len(prior) >= 5:
                cvd_now = pd.to_numeric(prior["cvd"], errors="coerce").iloc[-1]
                cvd_old = pd.to_numeric(prior["cvd"], errors="coerce").iloc[-5]
                if pd.notna(cvd_now) and pd.notna(cvd_old):
                    if cvd_now > cvd_old:
                        score += 2; reasons.append("GoCharting CVD rising")
                    elif cvd_now < cvd_old:
                        score -= 2; reasons.append("GoCharting CVD falling")

        recent = df5.tail(10)
        price_change = recent["Close"].iloc[-1] - recent["Close"].iloc[0]
        if of["source"] == "GOCHARTING_REAL":
            of_rows = real_of[real_of["Symbol"] == symbol].sort_values("DateTime")
            delta_series = pd.to_numeric(of_rows["delta"], errors="coerce").dropna()
            delta_mean = delta_series.tail(10).mean() if not delta_series.empty else np.nan
            if len(of_rows) >= 10:
                d0 = pd.to_numeric(of_rows["delta"], errors="coerce").iloc[-10]
                d1 = pd.to_numeric(of_rows["delta"], errors="coerce").iloc[-1]
                if pd.notna(d0) and pd.notna(d1):
                    if price_change > 0 and d1 < d0:
                        score -= 5
                        reasons.append("bearish price/real-Delta divergence")
                    elif price_change < 0 and d1 > d0:
                        score += 5
                        reasons.append("bullish price/real-Delta divergence")
        else:
            delta_mean = recent["DELTA_PROXY"].mean()
        if price_change > 0 and delta_mean < 0:
            score -= 4; reasons.append("price up while Delta proxy weakens")
        elif price_change < 0 and delta_mean > 0:
            score += 4; reasons.append("price down while Delta proxy strengthens")

        score = float(max(-100, min(100, score)))

        if score >= STRONG_THRESHOLD:
            signal = "STRONG BUY"
        elif score >= BUY_THRESHOLD:
            signal = "BUY"
        elif score <= -STRONG_THRESHOLD:
            signal = "STRONG SELL"
        elif score <= -SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "NO SIGNAL"

        price = float(r["Close"])
        atr_value = float(r["ATR"]) if pd.notna(r["ATR"]) else np.nan
        entry = price
        if pd.notna(atr_value) and atr_value > 0 and signal != "NO SIGNAL":
            if "BUY" in signal:
                stop = entry - 1.2 * atr_value
                target = entry + 2 * (entry - stop)
            else:
                stop = entry + 1.2 * atr_value
                target = entry - 2 * (stop - entry)
        else:
            stop = np.nan
            target = np.nan

        return {
            "Symbol": symbol, "Time": df5.index[-1], "Signal": signal,
            "Score": round(score, 1), "Price": round(price, 2),
            "Entry": round(entry, 2),
            "Stop Loss": round(stop, 2) if pd.notna(stop) else np.nan,
            "Target": round(target, 2) if pd.notna(target) else np.nan,
            "ATR": round(atr_value, 2) if pd.notna(atr_value) else np.nan,
            "VWAP": round(float(r["VWAP"]), 2) if pd.notna(r["VWAP"]) else np.nan,
            "RSI": round(float(r["RSI"]), 2) if pd.notna(r["RSI"]) else np.nan,
            "Volume x": round(float(r["VOL_RATIO"]), 2) if pd.notna(r["VOL_RATIO"]) else np.nan,
            "Delta %": round(float(of["delta_pct"]), 2) if pd.notna(of["delta_pct"]) else np.nan,
            "Order Flow": of["direction"], "Order Flow Source": of["source"],
            "Order Flow Score": of["score"],
            "CVD": round(float(of["cvd"]), 2) if pd.notna(of["cvd"]) else np.nan,
            "Approx POC": round(poc, 2) if pd.notna(poc) else np.nan,
            "Approx VAH": round(vah, 2) if pd.notna(vah) else np.nan,
            "Approx VAL": round(val, 2) if pd.notna(val) else np.nan,
            "CVD Change 5": round(float(of["cvd_change_5"]), 2) if pd.notna(of["cvd_change_5"]) else np.nan,
            "Max Delta": round(float(of["max_delta"]), 2) if pd.notna(of["max_delta"]) else np.nan,
            "Min Delta": round(float(of["min_delta"]), 2) if pd.notna(of["min_delta"]) else np.nan,
            "Extreme Delta Ratio": round(float(of["extreme_ratio"]), 2) if pd.notna(of["extreme_ratio"]) else np.nan,
            "Trades": round(float(of["trades"]), 0) if pd.notna(of["trades"]) else np.nan,
            "Reasons": " | ".join(reasons),
            "Warning": "OHLCV Delta proxy — not footprint Delta" if of["source"] == "OHLCV_PROXY" else "Genuine GoCharting order-flow supplied",
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}

with st.sidebar:
    st.header("Scanner Settings")
    interval = st.selectbox("Base timeframe", ["5m", "15m"], index=0)
    # Yahoo/yfinance limits intraday history to the most recent 60 days.
    # Keep the UI to periods that reliably work for 5m/15m data.
    period = st.selectbox("Data period", ["5d", "1mo"], index=1)
    workers = st.slider("Parallel workers", 2, 10, 6)
    buy_threshold = st.slider("BUY threshold", 50, 90, BUY_THRESHOLD)
    sell_threshold = st.slider("SELL threshold", 50, 90, SELL_THRESHOLD)
    strong_threshold = st.slider("Strong signal threshold", 70, 95, STRONG_THRESHOLD)
    real_required = st.checkbox("Require genuine GoCharting order flow", value=False)
    st.divider()
    st.markdown("GoCharting export: Time/DateTime + Delta required; Symbol, BuyVolume, SellVolume, MaxDelta, MinDelta and CVD are supported when present.")

uploaded = st.file_uploader(
    "Optional: upload GoCharting chart-data CSV export(s)",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload one combined CSV or multiple GoCharting exports. If Symbol is absent, the stock symbol is inferred from the filename."
)
real_of = read_orderflow(uploaded)

if not real_of.empty:
    st.success(f"Loaded {len(real_of):,} genuine order-flow rows for {real_of["Symbol"].nunique()} symbol(s).")
    with st.expander("Loaded GoCharting data preview"):
        st.dataframe(real_of.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("No usable GoCharting order-flow CSV loaded. The scanner will use an OHLCV Delta/CVD proxy and label it clearly.")

symbols, _ = get_nifty200()
st.write(f"**Nifty 200 symbols available:** {len(symbols)}")

run = st.button("🚀 Run Nifty 200 Scanner", type="primary", use_container_width=True)

@st.cache_data(ttl=300, show_spinner=False)
def download_market_batch(symbols, period, interval):
    """Download Nifty 200 intraday data in batches instead of 200 individual requests."""
    output = {}
    batch_size = 35
    for start in range(0, len(symbols), batch_size):
        batch_symbols = symbols[start:start + batch_size]
        tickers = [s + ".NS" for s in batch_symbols]
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=30,
            )
            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for symbol in batch_symbols:
                    ticker = symbol + ".NS"
                    try:
                        if ticker in data.columns.get_level_values(0):
                            output[symbol] = data[ticker].copy()
                        elif ticker in data.columns.get_level_values(1):
                            output[symbol] = data.xs(ticker, axis=1, level=1).copy()
                    except Exception:
                        continue
            elif len(batch_symbols) == 1:
                output[batch_symbols[0]] = data.copy()
        except Exception:
            continue
    return output


if run:
    progress = st.progress(0)
    status = st.empty()
    results = []
    errors = []

    status.write("Downloading Nifty 200 market data in batches…")
    downloaded = download_market_batch(symbols, period, interval)

    if not downloaded:
        st.error(
            "Yahoo Finance returned no usable intraday data. "
            "Try again in a few minutes, or use 15m + 5d/1mo. "
            "The app cannot create genuine order-flow data from Yahoo alone."
        )
        st.stop()

    st.info(f"Market data received for {len(downloaded)} of {len(symbols)} Nifty 200 symbols.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_symbol, s, period, interval, real_of, downloaded): s
            for s in symbols
            if s in downloaded
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            status.write(f"Calculating {symbol} — {i}/{total}")
            try:
                result = future.result()
                if "Error" in result:
                    errors.append(result)
                else:
                    results.append(result)
            except Exception as exc:
                errors.append({"Symbol": symbol, "Error": str(exc)})
            progress.progress(i / total)

    scanned_symbols = set(downloaded.keys())
    for symbol in symbols:
        if symbol not in scanned_symbols:
            errors.append({"Symbol": symbol, "Error": "No Yahoo intraday data returned"})

    progress.empty()
    status.empty()

    df = pd.DataFrame(results)
    if df.empty:
        st.error("No stock could be scanned. Check data connectivity.")
        st.stop()

    def final_signal(score):
        if score >= strong_threshold:
            return "STRONG BUY"
        if score >= buy_threshold:
            return "BUY"
        if score <= -strong_threshold:
            return "STRONG SELL"
        if score <= -sell_threshold:
            return "SELL"
        return "NO SIGNAL"

    df["Signal"] = df["Score"].apply(final_signal)
    if real_required:
        df.loc[df["Order Flow Source"] != "GOCHARTING_REAL", "Signal"] = "NO SIGNAL"

    final_df = df[df["Signal"].isin(["BUY", "STRONG BUY", "SELL", "STRONG SELL"])].copy()
    final_df["Rank"] = final_df["Score"].abs()
    final_df = final_df.sort_values(["Rank", "Score"], ascending=[False, False])
    candidates = df.assign(Rank=df["Score"].abs()).sort_values("Rank", ascending=False).head(50)

    st.session_state["scan_df"] = df
    st.session_state["final_df"] = final_df
    st.session_state["candidates"] = candidates
    st.session_state["errors"] = pd.DataFrame(errors)

if "final_df" in st.session_state:
    final_df = st.session_state["final_df"]
    candidates = st.session_state["candidates"]
    all_df = st.session_state["scan_df"]
    errors_df = st.session_state["errors"]

    st.subheader("🎯 Final BUY / SELL Signals")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", len(all_df))
    c2.metric("BUY signals", int(final_df["Signal"].str.contains("BUY").sum()))
    c3.metric("SELL signals", int(final_df["Signal"].str.contains("SELL").sum()))
    c4.metric("Data errors", len(errors_df))

    if final_df.empty:
        st.warning("No stock crossed the current final-signal threshold.")
    else:
        display = ["Symbol", "Signal", "Score", "Price", "Entry", "Stop Loss", "Target", "RSI", "Volume x", "Delta %", "Order Flow", "Order Flow Source", "CVD", "CVD Change 5", "Max Delta", "Min Delta", "Extreme Delta Ratio", "Approx POC", "Approx VAH", "Approx VAL"]
        st.dataframe(final_df[display], use_container_width=True, hide_index=True)

        st.subheader("🔎 Why each signal was produced")
        for _, row in final_df.iterrows():
            with st.expander(f"{row['Symbol']} — {row['Signal']} — Score {row['Score']}"):
                st.write(f"Price: {row['Price']} | Entry: {row['Entry']} | SL: {row['Stop Loss']} | Target: {row['Target']}")
                st.write(f"Order-flow source: {row['Order Flow Source']} | Delta %: {row['Delta %']} | CVD: {row['CVD']} | CVD Δ5: {row['CVD Change 5']}")
                st.write("Reason:")
                st.write(row["Reasons"])
                st.warning(row["Warning"])

    st.subheader("📋 Top Candidates")
    st.dataframe(candidates[["Symbol", "Signal", "Score", "Price", "RSI", "Volume x", "Delta %", "Order Flow", "Order Flow Source", "CVD Change 5", "Extreme Delta Ratio", "Approx POC", "Approx VAH", "Approx VAL"]], use_container_width=True, hide_index=True)

    st.subheader("📥 Download Results")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Final_Signals", index=False)
        candidates.to_excel(writer, sheet_name="Top_Candidates", index=False)
        all_df.to_excel(writer, sheet_name="All_Stocks", index=False)
        errors_df.to_excel(writer, sheet_name="Errors", index=False)

    st.download_button(
        "⬇️ Download Excel",
        data=excel_buffer.getvalue(),
        file_name="nifty200_orderflow_scanner.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Final Signals CSV",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name="nifty200_final_signals.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
with st.expander("ℹ️ How the final signal works"):
    st.markdown("""
### Signal engine
The score combines 5m, 15m and 60m trend, EMA20/50/200, VWAP, RSI, MACD, volume expansion, 20-bar breakout/breakdown, Delta, CVD and price/Delta disagreement.

### Genuine order flow
If GoCharting-derived data is uploaded, the app can use Delta, BuyVolume, SellVolume, MaxDelta, MinDelta and CVD.

### Important
The default OHLCV Delta is only a **pressure proxy**. It is not bid/ask footprint Delta. Genuine order-flow analysis requires actual bid/ask traded-volume data.
""")
