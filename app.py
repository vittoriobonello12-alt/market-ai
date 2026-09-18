import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from urllib.parse import quote_plus
from datetime import datetime, timezone
import re, math, time

app=FastAPI(title="Market AI V6")

CACHE={}
CACHE_TTL=600

POS={
"beat","beats","surge","surges","growth","grows","strong","stronger","profit","profits",
"record","upgrade","upgraded","approval","approved","partnership","deal","wins","win",
"raises","raised","positive","expands","expansion","buyback","dividend","outperform"
}
NEG={
"miss","misses","drop","drops","fall","falls","decline","declines","weak","weaker","loss",
"losses","downgrade","downgraded","lawsuit","investigation","recall","warning","cut","cuts",
"layoff","layoffs","negative","tariff","sanction","fine","fined","resign","resigns"
}
HIGH={
"bankruptcy","default","fraud","recall","acquisition","merger","takeover","guidance",
"earnings","regulator","sanction","tariff","lawsuit","investigation","ceo","fed","ecb",
"rate","rates","inflation","war","conflict","missile","election","tax"
}

def clean(s):
    return re.sub(r"\s+"," ",s or "").strip()

def rss(q, n=12):
    url="https://news.google.com/rss/search?q="+quote_plus(q)+"&hl=en-US&gl=US&ceid=US:en"
    req=Request(url,headers={"User-Agent":"Mozilla/5.0"})
    with urlopen(req,timeout=10) as r:
        root=ET.fromstring(r.read())
    out=[]
    for it in root.findall(".//item")[:n]:
        title=clean(it.findtext("title"))
        link=it.findtext("link") or ""
        pub=it.findtext("pubDate") or ""
        source=it.findtext("source") or ""
        desc=clean(re.sub("<.*?>"," ",it.findtext("description") or ""))
        if title: out.append({"title":title,"link":link,"published":pub,"source":source,"description":desc})
    return out

def age_weight(pub):
    try:
        from email.utils import parsedate_to_datetime
        dt=parsedate_to_datetime(pub)
        hours=max(0,(datetime.now(timezone.utc)-dt.astimezone(timezone.utc)).total_seconds()/3600)
        return math.exp(-hours/(48*2.2))
    except: return .45

def event_key(title):
    t=title.lower()
    t=re.sub(r"[^a-z0-9 ]"," ",t)
    words=[w for w in t.split() if len(w)>3]
    return " ".join(words[:8])

def analyze_news(item,ticker):
    text=(item["title"]+" "+item.get("description","")).lower()
    words=set(re.findall(r"[a-z]+",text))
    pos=len(words&POS); neg=len(words&NEG); high=len(words&HIGH)
    direct = ticker.lower() in text
    company_terms = {"earnings","guidance","ceo","dividend","buyback","merger","acquisition","lawsuit","recall","approval"}
    direct_terms=len(words&company_terms)
    macro_terms={"fed","ecb","inflation","rates","tariff","sanction","war","conflict","oil","recession"}
    macro=len(words&macro_terms)
    sentiment=(pos-neg)/max(1,pos+neg)
    relevance=22 + (32 if direct else 0) + min(28,direct_terms*7) + min(18,high*2)
    if macro and not direct: relevance=max(10,relevance-12)
    relevance=int(np.clip(relevance,5,100))
    importance=18 + min(38,high*6) + min(24,direct_terms*6) + min(12,macro*2)
    if direct: importance+=15
    recency=age_weight(item["published"])
    impact=int(np.clip(importance*0.65 + relevance*0.35,0,100)*recency + 8*(1-recency))
    impact=int(np.clip(impact,0,100))
    direction="positiva" if sentiment>.18 else "negativa" if sentiment<-.18 else "neutrale"
    if direction=="positiva": signed=impact
    elif direction=="negativa": signed=-impact
    else: signed=0
    if relevance<25:
        reason="La notizia appare poco collegata direttamente al titolo; l'effetto potenziale è quindi limitato."
    elif direction=="positiva":
        reason="La notizia contiene segnali potenzialmente favorevoli e presenta una rilevanza diretta o tematica per il titolo."
    elif direction=="negativa":
        reason="La notizia contiene segnali potenzialmente sfavorevoli e presenta una rilevanza diretta o tematica per il titolo."
    else:
        reason="La notizia è pertinente, ma non emerge una direzione sufficientemente netta per considerarla positiva o negativa."
    return {**item,"relevance":relevance,"importance":int(importance),"impact":impact,
            "direction":direction,"signed_impact":signed,"reason":reason}

def technical(ticker,horizon):
    d=yf.download(ticker,period="3y",interval="1d",auto_adjust=True,progress=False,threads=False,timeout=15)
    if d.empty: raise ValueError("Nessun dato trovato")
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.dropna(subset=["Close"]).copy()
    d["sma20"]=d.Close.rolling(20).mean()
    d["sma50"]=d.Close.rolling(50).mean()
    d["sma200"]=d.Close.rolling(200).mean()
    delta=d.Close.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    d["rsi"]=100-(100/(1+rs))
    d["ret1"]=d.Close.pct_change()
    d["vol20"]=d.ret1.rolling(20).std()
    d["volchg"]=d.Volume.pct_change()
    d=d.dropna()
    cols=["sma20","sma50","sma200","rsi","ret1","vol20","volchg"]
    X=d[cols].copy()
    y=(d.Close.shift(-horizon)>d.Close).astype(int)
    valid=y.notna()
    X=X[valid]; y=y[valid]
    if len(X)<300: raise ValueError("Dati insufficienti")
    X=X.iloc[-900:]; y=y.loc[X.index]
    split=max(200,int(len(X)*.8))
    model=RandomForestClassifier(n_estimators=180,max_depth=8,min_samples_leaf=8,
                                 class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    model.fit(X.iloc[:split],y.iloc[:split])
    proba=float(model.predict_proba(X.iloc[-1:].values)[0,1])
    # technical context score
    last=d.iloc[-1]
    score=0
    score += np.clip((last.Close/last.sma20-1)*800,-25,25)
    score += np.clip((last.Close/last.sma50-1)*500,-25,25)
    score += np.clip((last.rsi-50)*.5,-15,15)
    tech=int(np.clip(score,-100,100))
    return proba,tech,round(float(last.Close),2),round(float(last.rsi),1),len(d)

@app.get("/",response_class=HTMLResponse)
def home():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(500, f"index.html non trovato in {os.path.dirname(os.path.abspath(__file__))}")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health(): return {"ok":True}

@app.get("/analyze")
def analyze(ticker:str,horizon:int=3):
    ticker=ticker.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}",ticker): raise HTTPException(400,"Ticker non valido")
    horizon=int(np.clip(horizon,1,7))
    key=(ticker,horizon)
    now=time.time()
    if key in CACHE and now-CACHE[key]["time"]<CACHE_TTL: return CACHE[key]["data"]
    try:
        proba,tech,price,rsi,n=technical(ticker,horizon)
        company=rss(f'"{ticker}" stock',10)
        macro=rss('"Federal Reserve" OR ECB OR inflation OR "interest rates" OR tariffs OR geopolitics',10)
        raw=company+macro
        seen=set(); news=[]
        for x in raw:
            k=event_key(x["title"])
            if k in seen: continue
            seen.add(k); news.append(analyze_news(x,ticker))
        news=sorted(news,key=lambda x:(x["relevance"]*0.55+x["impact"]*0.45),reverse=True)[:12]
        if news:
            # Group influence: strongest unique events, diminishing returns for duplicates.
            weighted=[]
            for x in news:
                w=(x["relevance"]/100)*(.55+.45*age_weight(x["published"]))
                weighted.append(x["signed_impact"]*w)
            ext_signed=sum(weighted)/max(1,sum((x["relevance"]/100) for x in news))
            ext_abs=sum(abs(x["signed_impact"])*(x["relevance"]/100) for x in news)/max(1,sum(x["relevance"]/100 for x in news))
            external=int(np.clip(0.55*ext_abs+0.45*min(100,ext_abs+abs(ext_signed)*.35),0,100))
            # dampen normal news; high external requires multiple strong signals
            strong=sum(1 for x in news if x["relevance"]>=65 and x["impact"]>=55)
            if strong==0: external=min(external,45)
            elif strong==1: external=min(external,68)
            direction_bias=float(np.clip(ext_signed/100,-1,1))
        else:
            external=0; direction_bias=0
        final=float(np.clip(proba + direction_bias*0.08,0.02,0.98))
        signal="RIALZISTA" if final>=.58 else "RIBASSISTA" if final<=.42 else "NEUTRALE"
        result={"ticker":ticker,"horizon":horizon,"prob_up":round(final,4),"prob_down":round(1-final,4),
                "signal":signal,"external_impact":external,
                "external_label":"BASSO" if external<30 else "MODERATO" if external<55 else "ALTO" if external<75 else "ECCEZIONALE",
                "technical_factor":tech,"news_factor":round(direction_bias*100),
                "macro_factor":round(np.mean([x["signed_impact"] for x in news if ticker.lower() not in x["title"].lower()]) if news else 0),
                "price":price,"rsi":rsi,"data_points":n,"news":news,
                "note":"Punteggi descrittivi del modello: non sono probabilità certe né garanzie di rendimento."}
        CACHE[key]={"time":now,"data":result}
        return result
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get("/chart")
def chart(ticker:str):
    ticker=ticker.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}",ticker):
        raise HTTPException(400,"Ticker non valido")
    try:
        d=yf.download(ticker,period="1y",interval="1d",auto_adjust=True,progress=False,threads=False,timeout=15)
        if d.empty: raise ValueError("Nessun dato trovato")
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d=d.dropna(subset=["Close"]).copy()
        d["sma20"]=d.Close.rolling(20).mean()
        d["sma50"]=d.Close.rolling(50).mean()
        d["sma200"]=d.Close.rolling(200).mean()
        delta=d.Close.diff()
        gain=delta.clip(lower=0).rolling(14).mean()
        loss=(-delta.clip(upper=0)).rolling(14).mean()
        rs=gain/loss.replace(0,np.nan)
        d["rsi"]=100-(100/(1+rs))
        d=d.tail(260)
        rows=[]
        for idx,row in d.iterrows():
            rows.append({
                "date":idx.strftime("%d/%m/%y"),
                "close":round(float(row.Close),2),
                "sma20":None if pd.isna(row.sma20) else round(float(row.sma20),2),
                "sma50":None if pd.isna(row.sma50) else round(float(row.sma50),2),
                "sma200":None if pd.isna(row.sma200) else round(float(row.sma200),2),
                "rsi":None if pd.isna(row.rsi) else round(float(row.rsi),1)
            })
        return {"ticker":ticker,"rows":rows}
    except Exception as e:
        raise HTTPException(500,str(e))
