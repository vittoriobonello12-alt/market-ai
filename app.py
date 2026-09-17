from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Market AI V3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FEATURES = ["sma20_dist","sma50_dist","sma200_dist","rsi14","ret1","vol20","volchg"]

def rsi(s, n=14):
    d=s.diff()
    g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    rs=g/l.replace(0,np.nan)
    return 100-100/(1+rs)

def data(ticker):
    d=yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if d.empty: raise ValueError("Ticker non trovato o dati non disponibili.")
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d[["Close","Volume"]].dropna()
    d["sma20"]=d.Close.rolling(20).mean()
    d["sma50"]=d.Close.rolling(50).mean()
    d["sma200"]=d.Close.rolling(200).mean()
    d["rsi14"]=rsi(d.Close)
    d["ret1"]=d.Close.pct_change()
    d["vol20"]=d.ret1.rolling(20).std()
    d["volchg"]=d.Volume.pct_change().replace([np.inf,-np.inf],np.nan)
    d["sma20_dist"]=d.Close/d.sma20-1
    d["sma50_dist"]=d.Close/d.sma50-1
    d["sma200_dist"]=d.Close/d.sma200-1
    return d.replace([np.inf,-np.inf],np.nan).dropna()

def forecast(ticker, horizon):
    d=data(ticker)
    # Target is direction over the requested horizon.
    d["target"]=(d.Close.shift(-horizon)>d.Close).astype(int)
    train=d.iloc[:-horizon].dropna()
    if len(train)<500: raise ValueError("Dati storici insufficienti.")
    split=int(len(train)*.8)
    tr,te=train.iloc[:split],train.iloc[split:]
    m=RandomForestClassifier(n_estimators=350,max_depth=8,min_samples_leaf=10,
                             class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    m.fit(tr[FEATURES],tr.target)
    acc=float((m.predict(te[FEATURES])==te.target).mean())
    latest=d.iloc[[-1]]
    probs=m.predict_proba(latest[FEATURES])[0]
    classes=list(m.classes_)
    up=float(probs[classes.index(1)])
    down=float(probs[classes.index(0)])
    signal="RIALZISTA" if up>=.58 else ("RIBASSISTA" if up<=.42 else "NEUTRALE")
    # Last 60 sessions for the frontend chart.
    hist=d.tail(60)
    points=[{"date":str(i.date()),"close":round(float(r.Close),2),
             "sma50":round(float(r.sma50),2)} for i,r in hist.iterrows()]
    return {
        "ticker":ticker.upper(),"horizon_days":horizon,
        "date":str(latest.index[-1].date()),"price":float(latest.Close.iloc[0]),
        "sma20":float(latest.sma20.iloc[0]),"sma50":float(latest.sma50.iloc[0]),
        "sma200":float(latest.sma200.iloc[0]),"rsi14":float(latest.rsi14.iloc[0]),
        "vol20":float(latest.vol20.iloc[0]),"price_vs_sma50":float(latest.Close.iloc[0]/latest.sma50.iloc[0]-1),
        "up":up,"down":down,"signal":signal,"test_accuracy":acc,"points":points
    }

@app.get("/")
def home(): return FileResponse("index.html")

@app.get("/api/forecast/{ticker}")
def api_forecast(ticker:str, horizon:int=1):
    if horizon not in (1,3,7): raise HTTPException(400,"Orizzonte non valido.")
    try: return forecast(ticker.upper(),horizon)
    except Exception as e: raise HTTPException(400,str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="127.0.0.1",port=8000)
