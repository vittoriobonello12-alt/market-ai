from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Market AI V4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "sma20_dist", "sma50_dist", "sma200_dist",
    "rsi14", "ret1", "vol20", "volchg"
]

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _download(ticker):
    d = yf.download(
        ticker,
        period="3y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    if d is None or d.empty:
        raise ValueError("Ticker non trovato o dati non disponibili.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    needed = ["Close", "Volume"]
    if any(c not in d.columns for c in needed):
        raise ValueError("Yahoo Finance non ha restituito prezzo e volume.")
    d = d[needed].copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    d["sma20"] = d["Close"].rolling(20).mean()
    d["sma50"] = d["Close"].rolling(50).mean()
    d["sma200"] = d["Close"].rolling(200).mean()
    d["rsi14"] = rsi(d["Close"])
    d["ret1"] = d["Close"].pct_change()
    d["vol20"] = d["ret1"].rolling(20).std()
    d["volchg"] = d["Volume"].pct_change()
    d["sma20_dist"] = d["Close"] / d["sma20"] - 1
    d["sma50_dist"] = d["Close"] / d["sma50"] - 1
    d["sma200_dist"] = d["Close"] / d["sma200"] - 1

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 550:
        raise ValueError("Dati storici insufficienti per questo ticker.")
    return d

def forecast(ticker, horizon):
    d = _download(ticker)

    # Correctly exclude the final horizon rows: their future price is unknown.
    future_close = d["Close"].shift(-horizon)
    d["target"] = np.where(
        future_close.notna(),
        (future_close > d["Close"]).astype(int),
        np.nan
    )
    train = d.dropna(subset=FEATURES + ["target"]).copy()
    train["target"] = train["target"].astype(int)

    # Keep the model focused on recent history to reduce latency.
    train = train.tail(900)
    if len(train) < 500:
        raise ValueError("Dati storici insufficienti per addestrare il modello.")

    split = int(len(train) * 0.80)
    tr, te = train.iloc[:split], train.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(tr[FEATURES], tr["target"])
    accuracy = float((model.predict(te[FEATURES]) == te["target"]).mean())

    latest = d.iloc[-1:]
    probs = model.predict_proba(latest[FEATURES])[0]
    classes = list(model.classes_)
    up = float(probs[classes.index(1)])
    down = float(probs[classes.index(0)])

    signal = (
        "RIALZISTA" if up >= 0.58
        else "RIBASSISTA" if up <= 0.42
        else "NEUTRALE"
    )

    hist = d.tail(60)
    points = [
        {
            "date": str(i.date()),
            "close": round(float(row["Close"]), 2),
            "sma50": round(float(row["sma50"]), 2),
        }
        for i, row in hist.iterrows()
    ]

    return {
        "ticker": ticker.upper(),
        "horizon_days": horizon,
        "date": str(latest.index[-1].date()),
        "price": float(latest["Close"].iloc[0]),
        "sma20": float(latest["sma20"].iloc[0]),
        "sma50": float(latest["sma50"].iloc[0]),
        "sma200": float(latest["sma200"].iloc[0]),
        "rsi14": float(latest["rsi14"].iloc[0]),
        "vol20": float(latest["vol20"].iloc[0]),
        "price_vs_sma50": float(
            latest["Close"].iloc[0] / latest["sma50"].iloc[0] - 1
        ),
        "up": up,
        "down": down,
        "signal": signal,
        "test_accuracy": accuracy,
        "points": points,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

# Short in-memory cache: repeated taps don't retrain immediately.
_CACHE = {}
CACHE_SECONDS = 600

def cached_forecast(ticker, horizon):
    key = (ticker.upper(), horizon)
    now = datetime.now(timezone.utc).timestamp()
    item = _CACHE.get(key)
    if item and now - item[0] < CACHE_SECONDS:
        return item[1]
    result = forecast(ticker.upper(), horizon)
    _CACHE[key] = (now, result)
    return result

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/api/forecast/{ticker}")
def api_forecast(ticker: str, horizon: int = 1):
    if horizon not in (1, 3, 7):
        raise HTTPException(400, "Orizzonte non valido.")
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 15:
        raise HTTPException(400, "Inserisci un ticker valido.")
    try:
        return cached_forecast(ticker, horizon)
    except Exception as e:
        raise HTTPException(502, f"Analisi non disponibile: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}
