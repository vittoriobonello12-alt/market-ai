from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import re, time
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Market AI V5.1")
FEATURES = ["sma20_dist", "sma50_dist", "sma200_dist", "rsi14", "ret1", "vol20", "volchg"]
CACHE = {}
CACHE_SECONDS = 600

POS = {"growth","strong","surge","gain","record","profit","upgrade","bullish","agreement","deal","approved","approval","revenue","partnership","demand","positive","expands","launch","beat","beats"}
NEG = {"fall","drop","loss","downgrade","bearish","war","conflict","sanctions","tariff","tariffs","ban","restriction","crisis","miss","weak","decline","risk","recession","inflation","layoffs","lawsuit","probe","tension","cuts"}
HIGH = {"war","conflict","sanctions","tariff","tariffs","fed","ecb","interest rate","interest rates","inflation","recession","ban","election","crisis","emergency","default","bank failure"}
MEDIUM = {"earnings","guidance","forecast","acquisition","merger","antitrust","regulation","rates","jobs","unemployment","gdp","pce","cpi","revenue","profit","upgrade","downgrade","deal","partnership"}


def rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + (g / l.replace(0, np.nan)))


def market(t):
    d = yf.download(t, period="3y", interval="1d", auto_adjust=True, progress=False, threads=False, timeout=15)
    if d is None or d.empty:
        raise ValueError("Ticker non trovato o dati non disponibili.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    d["sma20"] = d.Close.rolling(20).mean(); d["sma50"] = d.Close.rolling(50).mean(); d["sma200"] = d.Close.rolling(200).mean()
    d["rsi14"] = rsi(d.Close); d["ret1"] = d.Close.pct_change(); d["vol20"] = d.ret1.rolling(20).std(); d["volchg"] = d.Volume.pct_change()
    d["sma20_dist"] = d.Close / d.sma20 - 1; d["sma50_dist"] = d.Close / d.sma50 - 1; d["sma200_dist"] = d.Close / d.sma200 - 1
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 550: raise ValueError("Dati storici insufficienti.")
    return d


def rss(q):
    u = "https://news.google.com/rss/search?q=" + quote_plus(q) + "&hl=en-US&gl=US&ceid=US:en"
    try:
        r = urlopen(Request(u, headers={"User-Agent": "Mozilla/5.0 MarketAI"}), timeout=8).read()
        root = ET.fromstring(r)
    except Exception:
        return []
    out = []
    for x in root.findall("./channel/item")[:8]:
        title = (x.findtext("title") or "").strip()
        if title:
            out.append({"title": title, "source": (x.findtext("source") or "News").strip(), "published": (x.findtext("pubDate") or "").strip()})
    return out


def word_score(text):
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(words & POS) - len(words & NEG)


def article_impact(title, is_company):
    s = title.lower()
    high_hits = sum(1 for k in HIGH if k in s)
    medium_hits = sum(1 for k in MEDIUM if k in s)
    sentiment_hits = abs(word_score(title))
    # Base relevance: an ordinary headline should stay low. High impact is earned by several strong signals.
    score = 0.05
    score += min(high_hits, 3) * 0.16
    score += min(medium_hits, 3) * 0.07
    score += min(sentiment_hits, 3) * 0.025
    if is_company and any(k in s for k in ("earnings", "guidance", "revenue", "profit", "acquisition", "merger")):
        score += 0.08
    return float(np.clip(score, 0, 0.95))


def news(t):
    company = rss(t + " stock")
    macro = rss("Federal Reserve OR ECB OR inflation OR interest rates OR tariffs OR geopolitics")
    alln = [{**x, "kind": "company"} for x in company[:6]] + [{**x, "kind": "macro"} for x in macro[:6]]
    if not alln:
        return {"available": False, "sentiment": 0, "impact": 0, "label": "NON DISPONIBILE", "articles": []}

    company_scores = [word_score(x["title"]) for x in company]
    macro_scores = [word_score(x["title"]) for x in macro]
    cs = sum(company_scores); ms = sum(macro_scores)
    sent = float(np.clip(0.7 * np.tanh(cs / 5) + 0.3 * np.tanh(ms / 5), -1, 1))

    impacts = [article_impact(x["title"], x["kind"] == "company") for x in alln]
    # Aggregate impact: recency is unavailable from every RSS feed in a consistent form, so
    # use the strongest headlines plus corroboration. One ordinary headline cannot create 90/100.
    strongest = sorted(impacts, reverse=True)
    corroboration = sum(1 for v in impacts if v >= 0.45)
    impact = 0.10 + 0.55 * strongest[0] + 0.10 * min(corroboration, 3) + 0.05 * min(len(alln), 6) / 6
    if strongest[0] < 0.35:
        impact = min(impact, 0.35)
    elif corroboration < 2:
        impact = min(impact, 0.65)
    impact = float(np.clip(impact, 0.02, 0.90))

    return {
        "available": True,
        "sentiment": sent,
        "impact": impact,
        "label": "POSITIVO" if sent > .15 else "NEGATIVO" if sent < -.15 else "NEUTRALE",
        "articles": alln[:10],
    }


def forecast(t, h):
    d = market(t)
    fut = d.Close.shift(-h)
    d["target"] = np.where(fut.notna(), (fut > d.Close).astype(int), np.nan)
    tr = d.dropna(subset=FEATURES + ["target"]).tail(900).copy(); tr.target = tr.target.astype(int)
    split = int(len(tr) * .8)
    model = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(tr[FEATURES].iloc[:split], tr.target.iloc[:split])
    acc = float((model.predict(tr[FEATURES].iloc[split:]) == tr.target.iloc[split:]).mean())
    p = model.predict_proba(d[FEATURES].iloc[-1:])[0]; cls = list(model.classes_); up = float(p[cls.index(1)])
    n = news(t)
    if n["available"]:
        up = float(np.clip(up + n["sentiment"] * .10 * max(n["impact"], .35), .02, .98))

    tech = float(np.clip((d.Close.iloc[-1] / d.sma50.iloc[-1] - 1) * 3, -1, 1))
    news_factor = float(n["sentiment"] if n["available"] else 0)
    macro_factor = float(n["sentiment"] * n["impact"] if n["available"] else 0)
    return {
        "ticker": t, "horizon_days": h, "date": str(d.index[-1].date()), "price": float(d.Close.iloc[-1]),
        "sma20": float(d.sma20.iloc[-1]), "sma50": float(d.sma50.iloc[-1]), "sma200": float(d.sma200.iloc[-1]),
        "rsi14": float(d.rsi14.iloc[-1]), "vol20": float(d.vol20.iloc[-1]), "price_vs_sma50": float(d.Close.iloc[-1] / d.sma50.iloc[-1] - 1),
        "up": up, "down": 1 - up, "signal": "RIALZISTA" if up >= .58 else "RIBASSISTA" if up <= .42 else "NEUTRALE",
        "test_accuracy": acc, "news": n,
        "external_impact": float(n["impact"] if n["available"] else 0),
        "factors": [{"name": "Tecnico", "value": tech}, {"name": "News", "value": news_factor}, {"name": "Macro / geopolitica", "value": macro_factor}],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


def cached(t, h):
    k = (t, h); now = time.time()
    if k in CACHE and now - CACHE[k][0] < CACHE_SECONDS:
        return CACHE[k][1]
    x = forecast(t, h); CACHE[k] = (now, x); return x


@app.get("/")
def home(): return FileResponse("index.html")

@app.get("/health")
def health(): return {"status": "ok", "version": "5.1"}

@app.get("/api/forecast/{ticker}")
def api(ticker: str, horizon: int = 1):
    if horizon not in (1, 3, 7): raise HTTPException(400, "Orizzonte non valido.")
    try: return cached(ticker.strip().upper(), horizon)
    except Exception as e: raise HTTPException(502, f"Analisi non disponibile: {e}")
