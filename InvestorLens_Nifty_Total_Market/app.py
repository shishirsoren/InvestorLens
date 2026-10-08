import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="InvestorLens — Indian Equity Intelligence",
    page_icon="📊",
    layout="wide",
)

# ----------------------------- Styling ---------------------------------
st.markdown(
    """
<style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    .hero {padding: 1.25rem 1.35rem; border-radius: 18px; background: linear-gradient(135deg, rgba(30,41,59,.96), rgba(15,23,42,.96)); border:1px solid rgba(148,163,184,.18); margin-bottom:1rem;}
    .hero h1 {font-size: 2.25rem; margin-bottom:.25rem;}
    .hero p {color:#cbd5e1; margin:0; font-size:1rem;}
    .tag {display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#1e293b; margin-right:.35rem; font-size:.78rem; color:#dbeafe; border:1px solid #334155;}
    .section-card {border:1px solid rgba(148,163,184,.16); border-radius:16px; padding:1rem; background:rgba(15,23,42,.48);}
    .small-muted {color:#94a3b8; font-size:.82rem;}
    .callout {padding:.8rem 1rem; border-radius:14px; border:1px solid rgba(148,163,184,.18); background:rgba(2,6,23,.32);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>📊 InvestorLens</h1>
  <p>5-year fundamentals + live/latest market data + valuation + leverage + CAPM + technicals + sentiment + alternative analysis.</p>
  <div style="margin-top:.7rem">
    <span class="tag">5Y Fundamentals</span><span class="tag">Live Quote</span><span class="tag">ROE Screener</span><span class="tag">CAPM</span><span class="tag">Technical</span><span class="tag">Sentiment</span><span class="tag">Piotroski</span><span class="tag">Altman</span><span class="tag">Beneish</span><span class="tag">SWOT</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Universe --------------------------------
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_universe() -> pd.DataFrame:
    """Load the current NIFTY Total Market constituent universe from NSE.

    NSE defines NIFTY Total Market as the broad-market universe combining
    NIFTY 500 and NIFTY Microcap 250. The constituent CSV is published by NSE
    and may contain ~750+ constituents after rebalancing/eligibility changes.
    A bundled CSV remains as an offline fallback only.
    """
    frames = []

    # 1) Official NSE constituent file (primary source)
    nse_url = "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
    try:
        r = requests.get(
            nse_url,
            headers={
                "User-Agent": "Mozilla/5.0 InvestorLens/1.0",
                "Accept": "text/csv,application/octet-stream,*/*",
                "Referer": "https://www.nseindia.com/",
            },
            timeout=15,
        )
        r.raise_for_status()
        raw = pd.read_csv(pd.io.common.BytesIO(r.content))
        if not raw.empty:
            frames.append(raw)
    except Exception:
        pass

    # 2) Optional additional universe URLs from environment
    urls = [u.strip() for u in os.getenv("UNIVERSE_URLS", "").split(",") if u.strip()]
    for url in urls:
        try:
            frames.append(pd.read_csv(url))
        except Exception:
            pass

    # 3) Bundled offline fallback
    local = os.path.join(os.path.dirname(__file__), "data", "universe.csv")
    if os.path.exists(local):
        try:
            frames.append(pd.read_csv(local))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["symbol", "company", "sector", "market_cap_cr", "universe"])

    normalized = []
    for raw in frames:
        df = raw.copy()
        # Normalize common NSE CSV column names without assuming one exact schema.
        aliases = {}
        for c in df.columns:
            k = str(c).strip().lower().replace(" ", "_").replace("-", "_")
            if k in {"symbol", "symbols", "security_symbol", "security_symbols", "nse_symbol"}:
                aliases[c] = "symbol"
            elif k in {"company_name", "company", "security", "name", "companyname"}:
                aliases[c] = "company"
            elif k in {"industry", "sector", "industry_name", "sector_name"}:
                aliases[c] = "sector"
            elif k in {"weight", "weight_percent", "weight_"}:
                aliases[c] = "weight_pct"
            elif k in {"isin", "isin_code"}:
                aliases[c] = "isin"
        df = df.rename(columns=aliases)
        for c in ["symbol", "company", "sector"]:
            if c not in df.columns:
                df[c] = ""
        if "market_cap_cr" not in df.columns:
            df["market_cap_cr"] = np.nan
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        # Remove non-equity/header rows.
        df = df[df["symbol"].ne("") & df["symbol"].ne("NAN")]
        df["company"] = df["company"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        df["market_cap_cr"] = pd.to_numeric(df["market_cap_cr"], errors="coerce")
        df["universe"] = "NIFTY TOTAL MARKET"
        normalized.append(df[[c for c in ["symbol", "company", "sector", "market_cap_cr", "weight_pct", "isin", "universe"] if c in df.columns]])

    df = pd.concat(normalized, ignore_index=True)
    # Prefer the first occurrence, which is the official NSE file when available.
    df = df.drop_duplicates("symbol", keep="first")
    return df.reset_index(drop=True)


universe = load_universe()

# Manual refresh for the official NSE constituent list. Useful after a semi-annual
# NIFTY reconstitution or when the app has been running for a while.
with st.sidebar:
    if st.button("🔄 Refresh NSE NIFTY universe"):
        load_universe.clear()
        st.rerun()

# Optional upload so user can load a verified current Top-1000 NSE list without changing source code.
with st.sidebar:
    st.header("Universe")
    st.caption("Primary universe: current NIFTY Total Market constituents from NSE")
    uploaded = st.file_uploader(
        "Optional: upload a custom NSE universe CSV",
        type=["csv"],
        help="Optional custom override. Columns accepted: symbol, company, sector, market_cap_cr.",
    )
    if uploaded is not None:
        try:
            up = pd.read_csv(uploaded)
            for c in ["symbol", "company", "sector"]:
                if c not in up.columns:
                    up[c] = ""
            if "market_cap_cr" not in up.columns:
                up["market_cap_cr"] = np.nan
            up["symbol"] = up["symbol"].astype(str).str.strip().str.upper()
            up["market_cap_cr"] = pd.to_numeric(up["market_cap_cr"], errors="coerce")
            universe = up.drop_duplicates("symbol").reset_index(drop=True)
            st.success(f"Loaded {len(universe):,} symbols")
        except Exception as e:
            st.error(f"CSV error: {e}")

if len(universe) < 500:
    st.warning(
        f"Only {len(universe):,} NIFTY Total Market constituents are currently loaded. "
        "The app will use the bundled offline fallback if NSE cannot be reached. "
        "Click '🔄 Refresh NSE NIFTY universe' when internet access to NSE is available."
    )
else:
    st.caption(f"Universe source: NSE NIFTY Total Market · {len(universe):,} constituents loaded")

# ----------------------------- Data adapters -----------------------------
def yf_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    if sym.startswith("^") or sym.endswith(".NS"):
        return sym
    return f"{sym}.NS"


@st.cache_data(ttl=60, show_spinner=False)
def get_live_quote(sym: str) -> Dict[str, Any]:
    """Use optional licensed/current quote API first, then Yahoo's latest available quote fields."""
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if key:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": yf_symbol(sym), "token": key},
                timeout=8,
            )
            q = r.json() if r.ok else {}
            if q.get("c"):
                return {
                    "price": float(q.get("c")),
                    "prev_close": float(q.get("pc")) if q.get("pc") is not None else np.nan,
                    "change": float(q.get("d")) if q.get("d") is not None else np.nan,
                    "change_pct": float(q.get("dp")) if q.get("dp") is not None else np.nan,
                    "high": float(q.get("h")) if q.get("h") is not None else np.nan,
                    "low": float(q.get("l")) if q.get("l") is not None else np.nan,
                    "open": float(q.get("o")) if q.get("o") is not None else np.nan,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Configured live quote API",
                }
        except Exception:
            pass

    try:
        t = yf.Ticker(yf_symbol(sym))
        fi = t.fast_info
        last = fi.get("last_price", fi.get("lastPrice"))
        prev = fi.get("previous_close", fi.get("previousClose"))
        if last is None:
            raise ValueError("No latest price")
        last = float(last)
        prev = float(prev) if prev is not None else np.nan
        ch = last - prev if np.isfinite(prev) else np.nan
        chp = ch / prev * 100 if np.isfinite(ch) and prev != 0 else np.nan
        return {
            "price": last,
            "prev_close": prev,
            "change": ch,
            "change_pct": chp,
            "high": np.nan,
            "low": np.nan,
            "open": np.nan,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Yahoo Finance latest available quote",
        }
    except Exception:
        return {"price": np.nan, "prev_close": np.nan, "change": np.nan, "change_pct": np.nan, "source": "Unavailable"}


@st.cache_data(ttl=900, show_spinner=False)
def get_history(sym: str, period: str = "5y") -> pd.DataFrame:
    t = yf.Ticker(yf_symbol(sym))
    df = t.history(period=period, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(sym: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(yf_symbol(sym)).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_financials(sym: str) -> Dict[str, pd.DataFrame]:
    t = yf.Ticker(yf_symbol(sym))
    out: Dict[str, pd.DataFrame] = {}
    for key, fn in [
        ("income", "income_stmt"),
        ("balance", "balance_sheet"),
        ("cashflow", "cashflow"),
        ("q_income", "quarterly_income_stmt"),
        ("q_balance", "quarterly_balance_sheet"),
        ("q_cashflow", "quarterly_cashflow"),
    ]:
        try:
            d = getattr(t, fn)
            out[key] = d if isinstance(d, pd.DataFrame) else pd.DataFrame()
        except Exception:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_dividends(sym: str) -> pd.Series:
    try:
        d = yf.Ticker(yf_symbol(sym)).dividends
        if d is None or len(d) == 0:
            return pd.Series(dtype=float)
        d = pd.to_numeric(d, errors="coerce").dropna()
        d.index = pd.to_datetime(d.index)
        return d.sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecasts(sym: str) -> Dict[str, Any]:
    t = yf.Ticker(yf_symbol(sym))
    out: Dict[str, Any] = {}
    for name in ["earnings_estimate", "revenue_estimate", "eps_trend", "growth_estimates", "analyst_price_targets"]:
        try:
            value = getattr(t, name)
            if callable(value):
                value = value()
            out[name] = value.copy() if isinstance(value, pd.DataFrame) else value
        except Exception:
            out[name] = {} if name == "analyst_price_targets" else pd.DataFrame()
    return out


def fiscal_year_label(dt: Any) -> str:
    """Indian financial year label based on period end date (Apr-Mar)."""
    d = pd.to_datetime(dt)
    end_year = int(d.year)
    start_year = end_year - 1 if d.month <= 3 else end_year
    return f"FY{start_year}-{str(end_year)[-2:]}"


def _fiscal_period_key(dt: Any) -> tuple:
    d = pd.to_datetime(dt)
    end_year = int(d.year) + (1 if d.month > 3 else 0)
    return end_year


def _normalize_annual_from_quarterly(df: pd.DataFrame, flow: bool) -> pd.DataFrame:
    """Aggregate quarterly provider data into Indian Apr-Mar financial years.
    Flow items are summed; balance-sheet items use the latest period in the FY.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    dates = sorted(_date_columns(df), key=lambda x: pd.to_datetime(x))
    if not dates:
        return pd.DataFrame()
    groups: Dict[int, List[Any]] = {}
    for d in dates:
        groups.setdefault(_fiscal_period_key(d), []).append(d)
    rows = {}
    for fy_end_year, fy_dates in sorted(groups.items()):
        row = {}
        for item in df.index:
            vals = pd.to_numeric(df.loc[item, fy_dates], errors="coerce").dropna()
            if not len(vals):
                row[item] = np.nan
            elif flow:
                row[item] = float(vals.sum())
            else:
                row[item] = float(vals.iloc[-1])
        rows[f"FY{fy_end_year-1}-{str(fy_end_year)[-2:]}"] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "Financial Year"
    return out


def build_statement_history(fin: Dict[str, pd.DataFrame], kind: str) -> pd.DataFrame:
    annual = fin.get(kind, pd.DataFrame())
    qkey = {"income":"q_income", "balance":"q_balance", "cashflow":"q_cashflow"}[kind]
    quarterly = fin.get(qkey, pd.DataFrame())
    flow = kind != "balance"

    def normalize_annual(df: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(_date_columns(df), key=lambda x: pd.to_datetime(x))
        if not dates:
            return pd.DataFrame()
        rows = {}
        for d in dates:
            rows[fiscal_year_label(d)] = pd.to_numeric(df[d], errors="coerce")
        out = pd.DataFrame(rows).T
        out.index.name = "Financial Year"
        # If duplicate periods exist, keep the latest column for that FY.
        out = out[~out.index.duplicated(keep="last")]
        return out.sort_index()

    a = normalize_annual(annual)
    q = _normalize_annual_from_quarterly(quarterly, flow)
    # Annual audited data is preferred. Quarterly-derived FYs only fill missing years.
    if not a.empty and not q.empty:
        a = a.combine_first(q)
    elif a.empty:
        a = q
    return a.tail(5)


def statement_to_crore(df: pd.DataFrame, rows: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    found=[]
    for wanted in rows:
        exact=[i for i in df.index if str(i).lower()==wanted.lower()]
        match=exact[0] if exact else next((i for i in df.index if wanted.lower() in str(i).lower()), None)
        if match is not None and match not in found:
            found.append(match)
    if not found:
        return pd.DataFrame()
    x=df.loc[found].copy()/1e7
    x.index=[str(i) for i in x.index]
    return x


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(sym: str, limit: int = 15) -> List[Dict[str, Any]]:
    rows = []
    try:
        t = yf.Ticker(yf_symbol(sym))
        items = t.news or []
        for n in items[:limit]:
            c = n.get("content", n)
            title = c.get("title") if isinstance(c, dict) else n.get("title")
            publisher = (
                c.get("provider", {}).get("displayName")
                if isinstance(c, dict) and isinstance(c.get("provider"), dict)
                else n.get("publisher")
            )
            ts = c.get("pubDate") if isinstance(c, dict) else n.get("providerPublishTime")
            link = ""
            if isinstance(c, dict):
                cu = c.get("canonicalUrl")
                if isinstance(cu, dict):
                    link = cu.get("url", "")
            rows.append({"title": title or "", "publisher": publisher or "", "published": ts, "link": link})
    except Exception:
        pass
    return [r for r in rows if r["title"]]


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty50(period: str = "5y") -> pd.DataFrame:
    return get_history("^NSEI", period=period)

# ----------------------------- Helpers ----------------------------------
def latest_col(df: pd.DataFrame, aliases: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    index_text = {str(c).lower(): c for c in df.index}
    for a in aliases:
        if a.lower() in index_text:
            return df.loc[index_text[a.lower()]]
    for a in aliases:
        for c in df.index:
            if a.lower() in str(c).lower():
                return df.loc[c]
    return None


def as_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def series_latest(df: pd.DataFrame, aliases: List[str]) -> float:
    s = latest_col(df, aliases)
    if s is None:
        return np.nan
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.iloc[0]) if len(vals) else np.nan


def last_two_values(df: pd.DataFrame, aliases: List[str]) -> Tuple[float, float]:
    s = latest_col(df, aliases)
    if s is None:
        return np.nan, np.nan
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan, np.nan
    return float(vals.iloc[0]), float(vals.iloc[1])


def growth(latest, prior):
    if not np.isfinite(latest) or not np.isfinite(prior) or prior == 0:
        return np.nan
    return (latest / prior - 1.0) * 100


def safe_div(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def fmt(v, decimals=2):
    return "N/A" if not np.isfinite(v) else f"{v:,.{decimals}f}"

# ----------------------------- 5-year financial history -----------------
def _date_columns(df: pd.DataFrame) -> List[Any]:
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        try:
            pd.to_datetime(c)
            cols.append(c)
        except Exception:
            continue
    return cols


def _annual_statement_frame(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    dates = _date_columns(df)
    if not dates:
        return pd.DataFrame()
    result = pd.DataFrame(index=[fiscal_year_label(d) for d in dates])
    result.index.name = "Financial Year"
    for label, al in aliases.items():
        s = latest_col(df, al)
        result[label] = [as_float(s.get(d, np.nan)) if s is not None else np.nan for d in dates]
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index().tail(5)


def _find_statement_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols={str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    for a in aliases:
        for c in df.columns:
            if a.lower() in str(c).lower():
                return c
    return None



# Canonical financial-statement aliases used throughout InvestorLens.
# Keep these broad enough to handle common yfinance/XBRL naming variants.
FIN_ALIASES = {
    "Revenue": ["Total Revenue", "Operating Revenue", "OperatingRevenue"],
    "Net Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
        "NetIncomeCommonStockholders",
    ],
    "Gross Profit": ["Gross Profit", "GrossProfit"],
    "Operating Income": ["Operating Income", "OperatingIncome"],
    "EBIT": ["EBIT", "Operating Income", "OperatingIncome"],
    "EBITDA": ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA"],
    "Assets": ["Total Assets", "TotalAssets"],
    "Equity": ["Stockholders Equity", "Stockholders' Equity", "Total Stockholder Equity", "StockholdersEquity"],
    "Debt": ["Total Debt", "TotalDebt"],
    "Cash": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "Current Assets": ["Current Assets", "CurrentAssets"],
    "Current Liabilities": ["Current Liabilities", "CurrentLiabilities"],
    "Liabilities": ["Total Liabilities Net Minority Interest", "Total Liab", "TotalLiabilitiesNetMinorityInterest"],
    "Receivables": ["Accounts Receivable", "Receivables", "AccountsReceivable"],
    "PPE": ["Net PPE", "Property Plant Equipment", "NetPPE"],
    "Retained Earnings": ["Retained Earnings", "Retained Earnings And Accumulated Deficit", "RetainedEarnings"],
    "CFO": ["Operating Cash Flow", "Total Cash From Operating Activities", "OperatingCashFlow"],
    "Capex": ["Capital Expenditure", "Capital Expenditure Reported", "CapitalExpenditure"],
    "Depreciation": ["Depreciation And Amortization", "Depreciation", "DepreciationAndAmortization"],
}

def build_5y_financial_history(fin: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces=[]
    maps=[
        ("income", {k:v for k,v in FIN_ALIASES.items() if k in {"Revenue","Net Income","Gross Profit","Operating Income","EBIT","EBITDA"}}),
        ("balance", {k:v for k,v in FIN_ALIASES.items() if k in {"Assets","Equity","Debt","Cash","Current Assets","Current Liabilities","Liabilities","Receivables","PPE","Retained Earnings"}}),
        ("cashflow", {k:v for k,v in FIN_ALIASES.items() if k in {"CFO","Capex","Depreciation"}}),
    ]
    for kind, amap in maps:
        df=build_statement_history(fin,kind)
        if df.empty: continue
        z=pd.DataFrame(index=df.index)
        for label,aliases in amap.items():
            col=_find_statement_column(df,aliases)
            z[label]=pd.to_numeric(df[col],errors="coerce") if col is not None else np.nan
        pieces.append(z)
    if not pieces: return pd.DataFrame()
    out=pd.concat(pieces,axis=1)
    out=out.loc[:,~out.columns.duplicated()]
    return out.sort_index().tail(5)

def history_ratios(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist
    h = hist.copy()
    h["Revenue Growth %"] = h["Revenue"].pct_change() * 100 if "Revenue" in h else np.nan
    h["Net Profit Growth %"] = h["Net Income"].pct_change() * 100 if "Net Income" in h else np.nan
    h["Net Margin %"] = h["Net Income"] / h["Revenue"] * 100 if {"Net Income","Revenue"}.issubset(h.columns) else np.nan
    if {"Net Income","Equity"}.issubset(h.columns):
        avg_eq = h["Equity"].rolling(2).mean()
        h["ROE %"] = h["Net Income"] / avg_eq * 100
    else:
        h["ROE %"] = np.nan
    if {"Net Income","Assets"}.issubset(h.columns):
        avg_assets = h["Assets"].rolling(2).mean()
        h["ROA %"] = h["Net Income"] / avg_assets * 100
    else:
        h["ROA %"] = np.nan
    h["Debt / Equity"] = h["Debt"] / h["Equity"] if {"Debt","Equity"}.issubset(h.columns) else np.nan
    h["FCF"] = h["CFO"] + h["Capex"] if {"CFO","Capex"}.issubset(h.columns) else np.nan
    return h

# ----------------------------- Ratios -----------------------------------
def calc_ratios(sym: str, fin: Dict[str, pd.DataFrame], info: Dict[str, Any], live_price: float = np.nan) -> Dict[str, float]:
    inc, bal, cf = fin.get("income", pd.DataFrame()), fin.get("balance", pd.DataFrame()), fin.get("cashflow", pd.DataFrame())
    revenue, revenue_prev = last_two_values(inc, FIN_ALIASES["Revenue"])
    net, net_prev = last_two_values(inc, FIN_ALIASES["Net Income"])
    ebit = series_latest(inc, FIN_ALIASES["EBIT"])
    ebitda = series_latest(inc, FIN_ALIASES["EBITDA"])
    interest = series_latest(inc, ["Interest Expense", "Interest Expense Non Operating"])
    gross = series_latest(inc, FIN_ALIASES["Gross Profit"])
    operating = series_latest(inc, FIN_ALIASES["Operating Income"])
    cur_assets = series_latest(bal, FIN_ALIASES["Current Assets"])
    cur_liab = series_latest(bal, FIN_ALIASES["Current Liabilities"])
    debt = series_latest(bal, FIN_ALIASES["Debt"])
    assets, assets_prev = last_two_values(bal, FIN_ALIASES["Assets"])
    eq, eq_prev = last_two_values(bal, FIN_ALIASES["Equity"])
    liabilities = series_latest(bal, FIN_ALIASES["Liabilities"])
    cash = series_latest(bal, FIN_ALIASES["Cash"])
    receivables = series_latest(bal, FIN_ALIASES["Receivables"])
    ppe = series_latest(bal, FIN_ALIASES["PPE"])
    retained = series_latest(bal, FIN_ALIASES["Retained Earnings"])
    dep = series_latest(cf, FIN_ALIASES["Depreciation"])
    cfo = series_latest(cf, FIN_ALIASES["CFO"])
    capex = series_latest(cf, FIN_ALIASES["Capex"])

    mcap = as_float(info.get("marketCap")) / 1e7 if np.isfinite(as_float(info.get("marketCap"))) else np.nan
    price = live_price if np.isfinite(live_price) else as_float(info.get("currentPrice"))
    shares = as_float(info.get("sharesOutstanding"))
    if not np.isfinite(mcap) and np.isfinite(price) and np.isfinite(shares):
        mcap = price * shares / 1e7

    avg_eq = np.nanmean([eq, eq_prev]) if np.any(np.isfinite([eq, eq_prev])) else np.nan
    avg_assets = np.nanmean([assets, assets_prev]) if np.any(np.isfinite([assets, assets_prev])) else np.nan

    r: Dict[str, float] = {}
    r["Revenue (₹ cr)"] = revenue / 1e7 if np.isfinite(revenue) else np.nan
    r["Revenue Growth %"] = growth(revenue, revenue_prev)
    r["Net Income (₹ cr)"] = net / 1e7 if np.isfinite(net) else np.nan
    r["Net Profit Growth %"] = growth(net, net_prev)
    r["Gross Margin %"] = safe_div(gross, revenue) * 100
    r["Operating Margin %"] = safe_div(operating, revenue) * 100
    r["Net Margin %"] = safe_div(net, revenue) * 100
    r["ROE %"] = safe_div(net, avg_eq) * 100
    r["ROA %"] = safe_div(net, avg_assets) * 100
    r["ROCE %"] = safe_div(ebit, (avg_assets - cur_liab)) * 100
    r["Current Ratio"] = safe_div(cur_assets, cur_liab)
    inventory = series_latest(bal, ["Inventory", "Inventory Net", "Inventories"])
    prepaid = series_latest(bal, ["Prepaid Assets", "Prepaid Expenses And Other Assets"])
    quick_assets = cur_assets - inventory if np.isfinite(inventory) else cur_assets
    if np.isfinite(prepaid):
        quick_assets -= prepaid
    r["Quick Ratio"] = safe_div(quick_assets, cur_liab)
    r["Debt / Equity"] = safe_div(debt, eq)
    r["Debt / Assets %"] = safe_div(debt, assets) * 100
    r["Net Debt (₹ cr)"] = (debt - cash) / 1e7 if np.all(np.isfinite([debt, cash])) else np.nan
    r["Interest Coverage"] = safe_div(ebit, abs(interest))
    r["Debt / EBITDA"] = safe_div(debt, ebitda)
    r["Asset Turnover"] = safe_div(revenue, avg_assets)
    r["Equity Multiplier"] = safe_div(avg_assets, avg_eq)
    r["CFO (₹ cr)"] = cfo / 1e7 if np.isfinite(cfo) else np.nan
    r["Capex (₹ cr)"] = capex / 1e7 if np.isfinite(capex) else np.nan
    r["FCF (₹ cr)"] = (cfo + capex) / 1e7 if np.all(np.isfinite([cfo, capex])) else np.nan
    r["Cash Conversion"] = safe_div(cfo, net)
    r["Market Cap (₹ cr)"] = mcap
    r["PE"] = as_float(info.get("trailingPE"))
    r["Forward PE"] = as_float(info.get("forwardPE"))
    r["PB"] = as_float(info.get("priceToBook"))
    r["EV/EBITDA"] = as_float(info.get("enterpriseToEbitda"))
    r["PEG"] = as_float(info.get("pegRatio"))
    dy = as_float(info.get("dividendYield", np.nan))
    pr = as_float(info.get("payoutRatio", np.nan))
    r["Dividend Yield %"] = dy * 100 if np.isfinite(dy) else np.nan
    r["Payout Ratio %"] = pr * 100 if np.isfinite(pr) else np.nan
    r["Beta (vendor)"] = as_float(info.get("beta"))
    r["_rev"] = revenue
    r["_rev_prev"] = revenue_prev
    r["_net"] = net
    r["_net_prev"] = net_prev
    r["_assets"] = assets
    r["_assets_prev"] = assets_prev
    r["_eq"] = eq
    r["_eq_prev"] = eq_prev
    r["_liab"] = liabilities
    r["_debt"] = debt
    r["_cash"] = cash
    r["_receivables"] = receivables
    r["_ppe"] = ppe
    r["_retained"] = retained
    r["_dep"] = dep
    r["_cfo"] = cfo
    r["_capex"] = capex
    r["_ebit"] = ebit
    r["_mcap"] = mcap * 1e7 if np.isfinite(mcap) else np.nan
    return r

# ----------------------------- Technical indicators --------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["Close"].astype(float)
    high = x["High"].astype(float)
    low = x["Low"].astype(float)
    x["SMA20"] = close.rolling(20).mean()
    x["SMA50"] = close.rolling(50).mean()
    x["SMA200"] = close.rolling(200).mean()
    x["EMA21"] = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["BBMid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    x["BBUpper"] = x["BBMid"] + 2 * std
    x["BBLower"] = x["BBMid"] - 2 * std
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    return x


def technical_summary(sym: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    df = add_indicators(get_history(sym, "2y"))
    if df.empty:
        return {"score": 0.5, "trend": "N/A"}, df
    row = df.iloc[-1]
    price = float(row["Close"])
    score = 0.5
    if np.isfinite(row.get("SMA50", np.nan)):
        score += 0.12 if price > row["SMA50"] else -0.12
    if np.isfinite(row.get("SMA200", np.nan)):
        score += 0.12 if price > row["SMA200"] else -0.12
    if np.isfinite(row.get("RSI14", np.nan)):
        if row["RSI14"] < 30:
            score += 0.08
        elif row["RSI14"] > 70:
            score -= 0.06
        elif row["RSI14"] > 50:
            score += 0.05
    if np.isfinite(row.get("MACD", np.nan)) and np.isfinite(row.get("MACDSignal", np.nan)):
        score += 0.08 if row["MACD"] > row["MACDSignal"] else -0.08
    score = float(np.clip(score, 0, 1))
    trend = "Bullish" if score >= 0.63 else "Bearish" if score <= 0.37 else "Mixed"
    atr_pct = safe_div(row["ATR14"], price) * 100
    tech = {
        "score": score,
        "trend": trend,
        "price": price,
        "RSI14": row["RSI14"],
        "SMA20": row["SMA20"],
        "SMA50": row["SMA50"],
        "SMA200": row["SMA200"],
        "MACD": row["MACD"],
        "MACDSignal": row["MACDSignal"],
        "BBUpper": row["BBUpper"],
        "BBLower": row["BBLower"],
        "ATR14": row["ATR14"],
        "ATR%": atr_pct,
    }
    return tech, df

# ----------------------------- Alternative analysis ---------------------
def piotroski(fin: Dict[str, pd.DataFrame]) -> Tuple[float, List[str]]:
    inc, bal, cf = fin.get("income", pd.DataFrame()), fin.get("balance", pd.DataFrame()), fin.get("cashflow", pd.DataFrame())
    y = 0
    notes = []
    net0, net1 = last_two_values(inc, FIN_ALIASES["Net Income"])
    cfo0, cfo1 = last_two_values(cf, FIN_ALIASES["CFO"])
    assets0, assets1 = last_two_values(bal, FIN_ALIASES["Assets"])
    debt0, debt1 = last_two_values(bal, FIN_ALIASES["Debt"])
    ca0, ca1 = last_two_values(bal, FIN_ALIASES["Current Assets"])
    cl0, cl1 = last_two_values(bal, FIN_ALIASES["Current Liabilities"])
    revenue0, revenue1 = last_two_values(inc, FIN_ALIASES["Revenue"])

    if np.isfinite(net0):
        y += int(net0 > 0); notes.append("Positive earnings" if net0 > 0 else "Negative earnings")
    if np.isfinite(cfo0):
        y += int(cfo0 > 0); notes.append("Positive operating cash flow" if cfo0 > 0 else "Negative operating cash flow")
    if np.all(np.isfinite([assets0, net0, assets1, net1, cfo0, cfo1])):
        roa0, roa1 = net0 / assets0, net1 / assets1
        y += int(roa0 > roa1); notes.append("ROA improving" if roa0 > roa1 else "ROA not improving")
        y += int(cfo0 > net0); notes.append("CFO exceeds net income" if cfo0 > net0 else "CFO does not exceed net income")
    if np.all(np.isfinite([debt0, debt1, assets0, assets1])):
        y += int(debt0 / assets0 < debt1 / assets1); notes.append("Leverage improved" if debt0 / assets0 < debt1 / assets1 else "Leverage not improved")
    if np.all(np.isfinite([ca0, cl0, ca1, cl1])):
        y += int(ca0 / cl0 > ca1 / cl1); notes.append("Liquidity improved" if ca0 / cl0 > ca1 / cl1 else "Liquidity not improved")
    if np.all(np.isfinite([revenue0, revenue1, net0, net1])):
        margin0, margin1 = safe_div(net0, revenue0), safe_div(net1, revenue1)
        y += int(margin0 > margin1); notes.append("Margin improved" if margin0 > margin1 else "Margin not improved")
    # Use asset turnover as the ninth check; share-count data is inconsistent across providers.
    if np.all(np.isfinite([revenue0, revenue1, assets0, assets1])):
        turnover0, turnover1 = revenue0 / assets0, revenue1 / assets1
        y += int(turnover0 > turnover1); notes.append("Asset turnover improved" if turnover0 > turnover1 else "Asset turnover not improved")
    return float(min(y, 9)), notes


def altman_z(r: Dict[str, float]) -> float:
    A = r.get("_assets", np.nan)
    B = r.get("_retained", np.nan)
    C = r.get("_ebit", np.nan)
    D = r.get("_mcap", np.nan)
    E = r.get("_liab", np.nan)
    revenue = r.get("_rev", np.nan)
    if not np.all(np.isfinite([A, B, C, D, E, revenue])) or A == 0 or E == 0:
        return np.nan
    return float(1.4 * (B / A) + 3.3 * (C / A) + 0.6 * (D / E) + 1.0 * (revenue / A))


def beneish_mscore(r: Dict[str, float], fin: Dict[str, pd.DataFrame]) -> float:
    inc, bal, cf = fin.get("income", pd.DataFrame()), fin.get("balance", pd.DataFrame()), fin.get("cashflow", pd.DataFrame())
    two = lambda df, aliases: last_two_values(df, aliases)
    rev0, rev1 = two(inc, FIN_ALIASES["Revenue"])
    rec0, rec1 = two(bal, FIN_ALIASES["Receivables"])
    gp0, gp1 = two(inc, FIN_ALIASES["Gross Profit"])
    ca0, ca1 = two(bal, FIN_ALIASES["Current Assets"])
    ppe0, ppe1 = two(bal, FIN_ALIASES["PPE"])
    dep0, dep1 = two(cf, FIN_ALIASES["Depreciation"])
    sga0, sga1 = two(inc, ["Selling General And Administration", "Selling General and Administrative"])
    ta0, ta1 = two(bal, FIN_ALIASES["Assets"])
    cl0, cl1 = two(bal, FIN_ALIASES["Current Liabilities"])
    ltd0, ltd1 = two(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    cfo0, cfo1 = two(cf, FIN_ALIASES["CFO"])
    net = r.get("_net", np.nan)
    vals = [rev0, rev1, rec0, rec1, gp0, gp1, ca0, ca1, ppe0, ppe1, dep0, dep1, sga0, sga1, ta0, ta1, cl0, cl1, ltd0, ltd1, cfo0, cfo1, net]
    if not all(np.isfinite(x) for x in vals):
        return np.nan
    try:
        dsri = (rec0 / rev0) / (rec1 / rev1)
        gmi = (gp1 / rev1) / (gp0 / rev0)
        aqi = (1 - (ca0 + ppe0) / ta0) / (1 - (ca1 + ppe1) / ta1)
        sgi = rev0 / rev1
        depi = (dep1 / (dep1 + ppe1)) / (dep0 / (dep0 + ppe0))
        sgai = (sga0 / rev0) / (sga1 / rev1)
        lvgi = ((ltd0 + cl0) / ta0) / ((ltd1 + cl1) / ta1)
        tata = (cfo0 - net) / ta0
        return float(-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    except Exception:
        return np.nan


def dupont(r: Dict[str, float]) -> float:
    avg_assets = np.nanmean([r.get("_assets", np.nan), r.get("_assets_prev", np.nan)])
    avg_equity = np.nanmean([r.get("_eq", np.nan), r.get("_eq_prev", np.nan)])
    a = safe_div(r.get("_net", np.nan), r.get("_rev", np.nan))
    b = safe_div(r.get("_rev", np.nan), avg_assets)
    c = safe_div(avg_assets, avg_equity)
    return a * b * c if np.all(np.isfinite([a, b, c])) else np.nan

# ----------------------------- CAPM -------------------------------------
def capm_analysis(sym: str, rf_pct: float, market_return_pct: float) -> Dict[str, float]:
    s = get_history(sym, "5y")
    m = get_nifty50("5y")
    if s.empty or m.empty:
        return {"beta": np.nan, "expected_return": np.nan, "stock_cagr": np.nan, "market_cagr": np.nan}
    sr = s["Close"].pct_change().dropna()
    mr = m["Close"].pct_change().dropna()
    joined = pd.concat([sr, mr], axis=1, join="inner").dropna()
    beta = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)[0, 1] / np.var(joined.iloc[:, 1], ddof=1) if len(joined) > 30 else np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else np.nan
    stock_cagr = ((s["Close"].iloc[-1] / s["Close"].iloc[0]) ** (1 / years) - 1) * 100 if np.isfinite(years) and years > 0 else np.nan
    market_years = (m.index[-1] - m.index[0]).days / 365.25 if len(m) > 1 else np.nan
    market_cagr = ((m["Close"].iloc[-1] / m["Close"].iloc[0]) ** (1 / market_years) - 1) * 100 if np.isfinite(market_years) and market_years > 0 else np.nan
    exp = rf_pct + beta * (market_return_pct - rf_pct) if np.isfinite(beta) else np.nan
    return {"beta": float(beta) if np.isfinite(beta) else np.nan, "expected_return": float(exp) if np.isfinite(exp) else np.nan, "stock_cagr": stock_cagr, "market_cagr": market_cagr}

# ----------------------------- Scoring ---------------------------------
def pct_score(x, good, bad, higher=True):
    if not np.isfinite(x):
        return 0.5
    if higher:
        if x >= good:
            return 1.0
        if x <= bad:
            return 0.0
    else:
        if x <= good:
            return 1.0
        if x >= bad:
            return 0.0
    return (x - bad) / (good - bad) if higher else (bad - x) / (bad - good)


def mean_or_default(values, default=0.5):
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else default


def research_score(r, tech: Dict[str, float], sentiment: float, pio: float, capm: Dict[str, float], financial_sector=False):
    fundamental = mean_or_default([
        pct_score(r.get("ROE %", np.nan), 18, 5, True),
        pct_score(r.get("ROCE %", np.nan), 20, 7, True),
        pct_score(r.get("Net Margin %", np.nan), 18, 3, True),
        pct_score(r.get("Revenue Growth %", np.nan), 15, -5, True),
    ])
    leverage = mean_or_default(([
        pct_score(r.get("ROE %", np.nan), 18, 5, True),
        pct_score(r.get("ROA %", np.nan), 2.0, 0.3, True),
        pct_score(r.get("Debt / Equity", np.nan), 4, 12, False),
    ] if financial_sector else [
        pct_score(r.get("Debt / Equity", np.nan), 0.5, 2.5, False),
        pct_score(r.get("Interest Coverage", np.nan), 8, 1.5, True),
        pct_score(r.get("Debt / EBITDA", np.nan), 2, 5, False),
    ]))
    valuation = mean_or_default([
        pct_score(r.get("PE", np.nan), 18, 45, False),
        pct_score(r.get("PB", np.nan), 3, 8, False),
        pct_score(r.get("EV/EBITDA", np.nan), 14, 28, False),
    ])
    tech_score = tech.get("score", 0.5)
    sent_score = 0.5 + 0.5 * max(-1, min(1, sentiment))
    quality = min(1, pio / 9) if np.isfinite(pio) else 0.5
    capm_score = 0.5 if not np.isfinite(capm.get("beta", np.nan)) else (0.65 if 0.7 <= capm["beta"] <= 1.3 else 0.5)
    score = 100 * (0.25 * fundamental + 0.15 * leverage + 0.15 * valuation + 0.15 * tech_score + 0.10 * sent_score + 0.15 * quality + 0.05 * capm_score)
    if score >= 72:
        signal = "Positive"
    elif score >= 55:
        signal = "Constructive / Watch"
    elif score >= 40:
        signal = "Neutral / Selective"
    else:
        signal = "Caution"
    return {"score": float(score), "signal": signal, "fundamental": fundamental, "leverage": leverage, "valuation": valuation, "technical": tech_score, "sentiment": sent_score, "quality": quality, "capm": capm_score}


def recommendation(score: Dict[str, Any], ratios: Dict[str, float], tech: Dict[str, float], pio: float, altz: float, beneish: float) -> Tuple[str, str, List[str], List[str]]:
    s = score["score"]
    drivers: List[str] = []
    cautions: List[str] = []
    if ratios.get("ROE %", np.nan) >= 20: drivers.append(f"ROE is {ratios['ROE %']:.1f}%")
    if ratios.get("Revenue Growth %", np.nan) >= 10: drivers.append(f"Revenue growth is {ratios['Revenue Growth %']:.1f}%")
    if ratios.get("Debt / Equity", np.nan) < 0.75: drivers.append(f"Debt/equity is relatively contained at {ratios['Debt / Equity']:.2f}")
    if ratios.get("FCF (₹ cr)", np.nan) > 0: drivers.append("Latest free cash flow is positive")
    if tech.get("trend") == "Bullish": drivers.append("Technical trend is bullish")
    if tech.get("trend") == "Bearish": cautions.append("Technical trend is bearish")
    if ratios.get("PE", np.nan) > 35: cautions.append(f"P/E is elevated at {ratios['PE']:.1f}")
    if ratios.get("Debt / Equity", np.nan) > 2: cautions.append(f"Debt/equity is elevated at {ratios['Debt / Equity']:.2f}")
    if np.isfinite(altz) and altz < 1.8: cautions.append("Altman screen is in the higher-risk zone")
    if np.isfinite(beneish) and beneish > -1.78: cautions.append("Beneish screen is above a common risk threshold")
    if np.isfinite(pio) and pio <= 3: cautions.append(f"Piotroski F-score is low at {pio:.0f}/9")

    if s >= 72:
        rec = "MODEL BUY / RESEARCH POSITIVE"
        detail = "The rules-based model shows a positive balance of profitability, valuation, quality, technical and risk signals."
    elif s >= 55:
        rec = "WATCH / SELECTIVE BUY"
        detail = "The model is constructive, but position sizing and entry valuation should be examined against the identified caveats."
    elif s >= 40:
        rec = "HOLD / NEUTRAL"
        detail = "The available signals are mixed; the stock merits further company-specific review rather than relying on a single factor."
    else:
        rec = "AVOID / HIGHER RISK SIGNAL"
        detail = "The rules-based model identifies more cautionary than supportive signals in the available data."
    return rec, detail, drivers[:6], cautions[:6]

# ----------------------------- SWOT ------------------------------------
def swot_from_data(r, tech, sentiment, pio, altman, beneish):
    strengths, weaknesses, opportunities, threats = [], [], [], []
    for label, key, good in [("ROE", "ROE %", 18), ("ROCE", "ROCE %", 20), ("Net margin", "Net Margin %", 18), ("Revenue growth", "Revenue Growth %", 10)]:
        v = r.get(key, np.nan)
        if np.isfinite(v) and v >= good:
            strengths.append(f"{label} is strong at {v:.1f}%")
        elif np.isfinite(v) and v < (good / 3):
            weaknesses.append(f"{label} is weak at {v:.1f}%")
    de = r.get("Debt / Equity", np.nan)
    if np.isfinite(de):
        (strengths if de < 0.6 else weaknesses if de > 2.0 else opportunities).append(f"Debt/equity is {de:.2f}")
    if tech.get("trend") == "Bullish": strengths.append("Price is above key moving averages")
    elif tech.get("trend") == "Bearish": threats.append("Price is below key moving averages")
    if sentiment > 0.2: opportunities.append("Recent news tone is positive")
    elif sentiment < -0.2: threats.append("Recent news tone is negative")
    if np.isfinite(pio) and pio >= 7: strengths.append(f"Piotroski F-score is strong ({pio:.0f}/9)")
    if np.isfinite(altman) and altman < 1.8: threats.append("Altman Z screen indicates elevated distress risk; sector applicability is limited")
    if np.isfinite(beneish) and beneish > -1.78: threats.append("Beneish M-score is above a common screening threshold; treat as a flag, not proof")
    if not opportunities: opportunities.append("Review industry tailwinds, product cycle, earnings catalysts and valuation")
    if not threats: threats.append("Monitor valuation, macro risk and company-specific execution")
    return {"Strengths": strengths[:5], "Weaknesses": weaknesses[:5], "Opportunities": opportunities[:5], "Threats": threats[:5]}

# ----------------------------- Screener engine --------------------------
def passes_filter(r: Dict[str, Any], params: Dict[str, float]) -> bool:
    checks = [
        ("PE", params["min_pe"], params["max_pe"], "between"),
        ("ROE %", params["min_roe"], None, "min"),
        ("Revenue Growth %", params["min_growth"], None, "min"),
        ("Debt / Equity", params["max_de"], None, "max"),
        ("ROCE %", params["min_roce"], None, "min"),
        ("FCF (₹ cr)", params["min_fcf"], None, "min"),
    ]
    for key, a, b, mode in checks:
        v = r.get(key, np.nan)
        if not np.isfinite(v):
            if params.get("require_data", True):
                return False
            continue
        if mode == "between" and not (a <= v <= b): return False
        if mode == "min" and v < a: return False
        if mode == "max" and v > a: return False
    return True


def scan_one(row: pd.Series, rf: float, market_return: float, params: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(row["symbol"]).upper()
    try:
        info = get_info(sym)
        fin = get_financials(sym)
        lq = get_live_quote(sym)
        ratios = calc_ratios(sym, fin, info, lq.get("price", np.nan))
        tech, _ = technical_summary(sym)
        news = get_news(sym, 8)
        sent = float(np.mean([SentimentIntensityAnalyzer().polarity_scores(x["title"])["compound"] for x in news])) if news else 0.0
        pio, _ = piotroski(fin)
        capm = capm_analysis(sym, rf, market_return)
        sector = str(info.get("sector") or row.get("sector", ""))
        financial_sector = any(k in sector.lower() for k in ["financial", "bank", "insurance"])
        rs = research_score(ratios, tech, sent, pio, capm, financial_sector)
        if not passes_filter(ratios, params):
            return {"Symbol": sym, "Company": info.get("longName") or row.get("company", sym), "Sector": sector, "_pass": False}
        if rs["score"] < params["min_score"]:
            return {"Symbol": sym, "Company": info.get("longName") or row.get("company", sym), "Sector": sector, "_pass": False}
        return {
            "Symbol": sym,
            "Company": info.get("longName") or row.get("company", sym),
            "Sector": sector,
            "Score": rs["score"],
            "Signal": rs["signal"],
            "ROE %": ratios.get("ROE %"),
            "ROCE %": ratios.get("ROCE %"),
            "Revenue Growth %": ratios.get("Revenue Growth %"),
            "P/E": ratios.get("PE"),
            "D/E": ratios.get("Debt / Equity"),
            "FCF (₹ cr)": ratios.get("FCF (₹ cr)"),
            "Technical": tech.get("trend"),
            "Sentiment": sent,
            "Price": lq.get("price", np.nan),
            "_pass": True,
        }
    except Exception as exc:
        return {"Symbol": sym, "Company": row.get("company", sym), "Sector": row.get("sector", ""), "_pass": False, "Error": str(exc)[:80]}


# ----------------------------- Filters ----------------------------------
with st.sidebar:
    st.subheader("Filters")
    min_mcap, max_mcap = st.slider("Market cap (₹ cr)", 0, 2_000_000, (0, 2_000_000), step=5_000)
    sectors = sorted([x for x in universe["sector"].dropna().unique().tolist() if str(x).strip()])
    selected_sectors = st.multiselect("Sector", sectors)
    min_roe = st.number_input("Minimum ROE %", value=0.0, step=1.0, help="Example: enter 20 for ROE > 20%")
    min_roce = st.number_input("Minimum ROCE %", value=-100.0, step=1.0)
    min_growth = st.number_input("Minimum revenue growth %", value=-100.0, step=5.0)
    min_fcf = st.number_input("Minimum FCF (₹ cr)", value=-1e12, step=100.0, format="%.0f")
    max_de = st.number_input("Maximum debt/equity", value=10.0, min_value=0.0, step=0.1)
    min_pe, max_pe = st.slider("P/E range", 0.0, 150.0, (0.0, 150.0), step=1.0)
    min_score = st.slider("Minimum model score", 0, 100, 0)
    require_data = st.checkbox("Require data for every selected filter", value=True)
    st.divider()
    st.subheader("Live data")
    auto_refresh = st.checkbox("Auto-refresh selected quote", value=True)
    refresh_seconds = st.slider("Refresh interval (seconds)", 30, 300, 60, step=30, disabled=not auto_refresh)

filtered = universe.copy()
if len(filtered):
    filtered = filtered[filtered["market_cap_cr"].fillna(0).between(min_mcap, max_mcap)]
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]

st.caption(
    f"Universe loaded: **{len(universe):,}** symbols · metadata-filtered pool: **{len(filtered):,}**. "
    "The ROE/ROCE/FCF/valuation filters are applied during the financial scan, so the screener can find companies such as ROE > 20% instead of filtering only on static universe metadata."
)

# ----------------------------- Company selection ------------------------
search = st.text_input("Search company or NSE symbol", placeholder="e.g. RELIANCE, TCS, HDFC BANK")
pool = filtered
if search:
    q = search.lower()
    pool = filtered[filtered.apply(lambda r: q in f"{r['symbol']} {r['company']}".lower(), axis=1)]

left, right = st.columns([2, 1])
with left:
    if len(pool):
        options = [f"{r.symbol} — {r.company}" for _, r in pool.head(250).iterrows()]
        selected_label = st.selectbox("Company", options)
        sym = selected_label.split(" — ")[0].strip()
    else:
        sym = st.text_input("NSE symbol", value="RELIANCE").upper().strip()
with right:
    rf = st.number_input("Risk-free rate %", value=float(os.getenv("DEFAULT_RISK_FREE_RATE", 6.0)), step=0.25)
    market_return = st.number_input("Market return assumption %", value=float(os.getenv("DEFAULT_MARKET_RETURN", 12.0)), step=0.25)

# ----------------------------- Load company -----------------------------
with st.spinner("Fetching 5-year data, market quote, financials and news…"):
    info = get_info(sym)
    fin = get_financials(sym)
    dividends = get_dividends(sym)
    forecasts = get_forecasts(sym)
    live = get_live_quote(sym)
    ratios = calc_ratios(sym, fin, info, live.get("price", np.nan))
    hist = build_5y_financial_history(fin)
    hist_r = history_ratios(hist)
    tech, tech_df = technical_summary(sym)
    sent_rows = get_news(sym)

analyzer = SentimentIntensityAnalyzer()
for row in sent_rows:
    row["sentiment"] = analyzer.polarity_scores(row["title"])["compound"]
sentiment = float(np.mean([r["sentiment"] for r in sent_rows])) if sent_rows else 0.0

pio, pio_notes = piotroski(fin)
altz = altman_z(ratios)
beneish = beneish_mscore(ratios, fin)
dup = dupont(ratios)
capm = capm_analysis(sym, rf, market_return)

univ_match = universe.loc[universe["symbol"] == sym]
sector = str(info.get("sector") or (univ_match["sector"].iloc[0] if len(univ_match) else ""))
financial_sector = any(k in sector.lower() for k in ["financial", "bank", "insurance"])
score = research_score(ratios, tech, sentiment, pio, capm, financial_sector)
rec, rec_detail, rec_drivers, rec_cautions = recommendation(score, ratios, tech, pio, altz, beneish)
swot = swot_from_data(ratios, tech, sentiment, pio, altz, beneish)
name = info.get("longName") or info.get("shortName") or (univ_match["company"].iloc[0] if len(univ_match) else sym)

# ----------------------------- Header metrics ---------------------------
st.subheader(f"{name} ({sym})")
st.caption(f"Sector: {sector or 'N/A'} · Quote source: {live.get('source','Unavailable')} · Quote checked: {live.get('timestamp','N/A')}")
st.caption("Data integrity: statement amounts are normalized to ₹ crore only for display; EPS and dividend/share remain ₹/share. Historical statements are not real-time market data. Exchange-grade real-time quotes require a licensed NSE/BSE/authorized feed.")

m = st.columns(7)
m[0].metric("LTP", f"₹{fmt(live.get('price', np.nan), 2)}")
m[1].metric("1D", f"{live.get('change_pct', np.nan):+.2f}%" if np.isfinite(live.get("change_pct", np.nan)) else "N/A")
m[2].metric("Market Cap", f"₹{fmt(ratios.get('Market Cap (₹ cr)', np.nan), 0)} cr")
m[3].metric("P/E", fmt(ratios.get("PE", np.nan), 1))
m[4].metric("ROE", f"{fmt(ratios.get('ROE %', np.nan), 1)}%")
m[5].metric("D/E", fmt(ratios.get("Debt / Equity", np.nan), 2))
m[6].metric("Model Score", f"{score['score']:.0f}/100")

# Live quote fragment; current/latest quote updates without reloading all analysis.
run_every = refresh_seconds if auto_refresh else None

@st.fragment(run_every=run_every)
def live_panel():
    q = get_live_quote(sym)
    if np.isfinite(q.get("price", np.nan)):
        st.markdown(
            f"<div class='callout'><b>Live quote</b> · ₹{q['price']:,.2f} · {q.get('change_pct', np.nan):+.2f}% vs previous close · {q.get('source','')}</div>",
            unsafe_allow_html=True,
        )

live_panel()

# ----------------------------- Tabs -------------------------------------
tabs = st.tabs([
    "⭐ Recommendation",
    "📌 Dashboard",
    "📊 5Y Financials",
    "📄 Income Statement",
    "💵 Cash Flow",
    "💰 Dividends",
    "🔮 Forecast",
    "📐 Ratios",
    "🏦 Leverage",
    "💰 CAPM",
    "📈 Technical",
    "📰 Sentiment",
    "🧪 Alternative",
    "🧭 SWOT",
    "🧰 Screener",
    "🔍 Data Audit",
])

with tabs[0]:
    st.subheader("Model recommendation")
    st.markdown(f"## {rec}")
    st.write(rec_detail)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model score", f"{score['score']:.0f}/100")
    c2.metric("Fundamental component", f"{score['fundamental']*100:.0f}/100")
    c3.metric("Technical component", f"{score['technical']*100:.0f}/100")
    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("### Supporting drivers")
        if rec_drivers:
            for x in rec_drivers: st.write("✅ " + x)
        else:
            st.write("No strong positive driver met the model checks.")
    with d2:
        st.markdown("### Risk / caution flags")
        if rec_cautions:
            for x in rec_cautions: st.write("⚠️ " + x)
        else:
            st.write("No major caution flag was generated by the current rules.")
    st.info("This recommendation is a rules-based research signal, not personalized investment advice or a guarantee of future returns. Validate financial statements, disclosures, valuation and your own risk tolerance before acting.")

with tabs[1]:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Composite research signal")
        st.progress(score["score"] / 100)
        st.markdown(f"### {score['signal']} · {score['score']:.0f}/100")
        cols = st.columns(7)
        labels = [("Fundamental", score["fundamental"]), ("Leverage", score["leverage"]), ("Valuation", score["valuation"]), ("Technical", score["technical"]), ("Sentiment", score["sentiment"]), ("Quality", score["quality"]), ("CAPM", score["capm"])]
        for c, (lab, val) in zip(cols, labels): c.metric(lab, f"{val*100:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Key investor checks")
        checks = {
            "Revenue Growth %": ratios.get("Revenue Growth %", np.nan),
            "Net Profit Growth %": ratios.get("Net Profit Growth %", np.nan),
            "ROCE %": ratios.get("ROCE %", np.nan),
            "Interest Coverage": ratios.get("Interest Coverage", np.nan),
            "FCF (₹ cr)": ratios.get("FCF (₹ cr)", np.nan),
            "Dividend Yield %": ratios.get("Dividend Yield %", np.nan),
        }
        st.dataframe(pd.DataFrame(checks, index=["Value"]).T, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("5-year price trend")
    five_price = get_history(sym, "5y")
    if not five_price.empty:
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=five_price.index, y=five_price["Close"], name="Close", line=dict(width=2)))
        chart.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title="Price (₹)")
        st.plotly_chart(chart, use_container_width=True)

with tabs[2]:
    st.subheader("5-year financial trend")
    if hist_r.empty:
        st.warning("The provider did not return enough financial statement history for this company.")
    else:
        display_cols = [c for c in ["Revenue","Net Income","CFO","Capex","FCF","ROE %","ROA %","Net Margin %","Debt / Equity"] if c in hist_r.columns]
        show = hist_r[display_cols].copy()
        for c in ["Revenue","Net Income","CFO","Capex","FCF"]:
            if c in show.columns: show[c] = show[c] / 1e7
        st.caption("All monetary statement figures are normalized to ₹ crore from provider-reported INR amounts. Flow items are summed by year when quarterly data is used; balance-sheet items use the latest period. Signs are preserved, so capex is normally negative and FCF = CFO + Capex.")
        st.dataframe(show.round(2), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Revenue" in hist_r.columns and "Net Income" in hist_r.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=hist_r.index, y=hist_r["Revenue"]/1e7, name="Revenue (₹ cr)"))
                fig.add_trace(go.Bar(x=hist_r.index, y=hist_r["Net Income"]/1e7, name="Net Income (₹ cr)"))
                fig.update_layout(height=360, barmode="group", margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "ROE %" in hist_r.columns and "Debt / Equity" in hist_r.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_r.index, y=hist_r["ROE %"], mode="lines+markers", name="ROE %"))
                fig.add_trace(go.Scatter(x=hist_r.index, y=hist_r["Debt / Equity"], mode="lines+markers", name="Debt / Equity", yaxis="y2"))
                fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), yaxis=dict(title="ROE %"), yaxis2=dict(title="Debt / Equity", overlaying="y", side="right"))
                st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Income Statement — 5 Financial Years")
    inc_hist=build_statement_history(fin,"income")
    if inc_hist.empty:
        st.warning("No income-statement history was returned by the configured provider.")
    else:
        wanted = ["Total Revenue","Operating Revenue","Cost Of Revenue","Gross Profit","Operating Expense","Operating Income","EBIT","EBITDA","Pretax Income","Tax Provision","Net Income","Net Income Common Stockholders","Diluted EPS","Basic EPS"]
        data={}
        for wanted_row in wanted:
            col=_find_statement_column(inc_hist,[wanted_row])
            if col is not None:
                data[wanted_row]=pd.to_numeric(inc_hist[col],errors="coerce")
        view=pd.DataFrame(data).T
        eps_rows=[r for r in ["Diluted EPS","Basic EPS"] if r in view.index]
        money_rows=[r for r in view.index if r not in eps_rows]
        if money_rows: view.loc[money_rows]=view.loc[money_rows]/1e7
        st.caption("Indian financial years (April–March). ₹ crore for statement amounts; EPS is ₹/share. Audited annual provider data is preferred; quarterly data is used only to fill missing FYs.")
        st.dataframe(view.round(2),use_container_width=True)
        st.download_button("Download 5-year income statement CSV",view.to_csv().encode(),file_name=f"{sym}_income_statement_5FY.csv",mime="text/csv")
        st.info(f"Periods available: {', '.join(map(str, inc_hist.index))}. If fewer than five periods appear, the configured provider did not supply enough historical statement periods; the app will not invent values.")

with tabs[4]:
    st.subheader("Cash Flow Statement — 5 Financial Years")
    cf_hist=build_statement_history(fin,"cashflow")
    if cf_hist.empty: st.warning("No cash-flow history was returned by the configured provider.")
    else:
        wanted=["Operating Cash Flow","Total Cash From Operating Activities","Capital Expenditure","Capital Expenditure Reported","Investing Cash Flow","Financing Cash Flow","Free Cash Flow","Depreciation And Amortization","Repurchase Of Capital Stock","Repayment Of Debt","Issuance Of Debt","Dividends Paid"]
        data={}
        for wanted_row in wanted:
            col=_find_statement_column(cf_hist,[wanted_row])
            if col is not None: data[wanted_row]=pd.to_numeric(cf_hist[col],errors="coerce")
        view=pd.DataFrame(data).T/1e7 if data else pd.DataFrame()
        st.caption("Indian financial years (April–March). ₹ crore; cash-flow signs are preserved. FCF should be interpreted after checking the provider's capex sign convention.")
        st.dataframe(view.round(2),use_container_width=True)
        st.download_button("Download 5-year cash flow CSV",view.to_csv().encode(),file_name=f"{sym}_cash_flow_5FY.csv",mime="text/csv")

with tabs[5]:
    st.subheader("Dividend history")
    if dividends.empty: st.warning("No dividend history was returned.")
    else:
        d=dividends.rename("Dividend per share (₹)").to_frame()
        annual=d.assign(Year=d.index.year).groupby("Year")["Dividend per share (₹)"].sum().tail(10).to_frame()
        c1,c2,c3=st.columns(3)
        y=annual.index.max()
        c1.metric("Latest annual dividend/share",f"₹{annual.loc[y].iloc[0]:.2f}")
        c2.metric("Dividend yield",f"{fmt(ratios.get('Dividend Yield %',np.nan),2)}%")
        c3.metric("Payout ratio",f"{fmt(ratios.get('Payout Ratio %',np.nan),2)}%")
        st.dataframe(annual.round(4),use_container_width=True)
        fig=go.Figure(go.Bar(x=annual.index,y=annual.iloc[:,0],name="Dividend/share")); fig.update_layout(height=350,yaxis_title="₹/share",margin=dict(l=10,r=10,t=20,b=10)); st.plotly_chart(fig,use_container_width=True)
        st.caption("Provider-reported cash distributions per share, summed by calendar year. Verify record/ex-date and corporate-action details before relying on payout figures.")

with tabs[6]:
    st.subheader("Forecast & analyst estimates")
    targets=forecasts.get("analyst_price_targets") or {}
    if isinstance(targets,dict) and targets:
        c=st.columns(4)
        for col,key,label in [(c[0],"current","Current target basis"),(c[1],"low","Low target"),(c[2],"mean","Mean target"),(c[3],"high","High target")]:
            v=as_float(targets.get(key)); col.metric(label,f"₹{v:,.2f}" if np.isfinite(v) else "N/A")
    for title,key in [("EPS estimates","earnings_estimate"),("Revenue estimates","revenue_estimate"),("EPS trend revisions","eps_trend"),("Growth estimates","growth_estimates")]:
        st.markdown(f"### {title}")
        obj=forecasts.get(key)
        if isinstance(obj,pd.DataFrame) and not obj.empty: st.dataframe(obj,use_container_width=True)
        else: st.info("No provider estimate table was returned.")
    st.warning("Forecasts are third-party/provider estimates, not guaranteed outcomes. Historical and forecast periods must not be mixed when interpreting ratios.")

with tabs[7]:
    st.subheader("Current financial ratio matrix")
    ratio_display = {k: v for k, v in ratios.items() if not k.startswith("_")}
    rd = pd.DataFrame({"Metric": list(ratio_display.keys()), "Value": list(ratio_display.values())})
    st.dataframe(rd, use_container_width=True, hide_index=True)
    st.info("Ratios use the latest provider-reported statement period. ROE uses average beginning/ending equity when available; ROCE uses EBIT / (average assets − current liabilities); Quick Ratio excludes inventory and prepaid/current non-quick assets when available. Banks/insurers require sector-specific interpretation.")

with tabs[8]:
    st.subheader("Leverage & solvency")
    leverage_items = ["Debt / Equity","Debt / Assets %","Net Debt (₹ cr)","Interest Coverage","Debt / EBITDA","Current Ratio","Quick Ratio","Cash Conversion"]
    ld = pd.DataFrame({"Leverage Metric": leverage_items, "Value": [ratios.get(x, np.nan) for x in leverage_items]})
    st.dataframe(ld, use_container_width=True, hide_index=True)
    st.caption("Leverage interpretation is business-model dependent; financial firms need sector-specific metrics.")

with tabs[9]:
    st.subheader("CAPM / systematic-risk view")
    c = st.columns(5)
    c[0].metric("Beta", fmt(capm["beta"], 2))
    c[1].metric("CAPM expected return", f"{fmt(capm['expected_return'], 2)}%")
    c[2].metric("5Y stock CAGR", f"{fmt(capm['stock_cagr'], 2)}%")
    c[3].metric("5Y NIFTY CAGR", f"{fmt(capm['market_cagr'], 2)}%")
    c[4].metric("Risk-free / Market", f"{rf:.2f}% / {market_return:.2f}%")
    st.write("CAPM expected return = Risk-free rate + Beta × (Market return − Risk-free rate). Beta is calculated from daily stock and NIFTY 50 returns over the available 5-year history.")

with tabs[10]:
    st.subheader("Technical analysis")
    tcols = st.columns(7)
    for c, (lab, key) in zip(tcols, [("Trend","trend"),("RSI","RSI14"),("SMA20","SMA20"),("SMA50","SMA50"),("SMA200","SMA200"),("MACD","MACD"),("ATR %","ATR%")]):
        val = tech.get(key, np.nan)
        disp = val if isinstance(val, str) else (f"{val:.2f}" if np.isfinite(val) else "N/A")
        c.metric(lab, disp)
    if not tech_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=tech_df.index, open=tech_df["Open"], high=tech_df["High"], low=tech_df["Low"], close=tech_df["Close"], name="OHLC"))
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA50"], name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA200"], name="SMA200", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["BBUpper"], name="BB Upper", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["BBLower"], name="BB Lower", line=dict(width=1)))
        fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

with tabs[11]:
    st.subheader("News sentiment")
    st.metric("Average headline sentiment", f"{sentiment:+.2f}")
    if sent_rows:
        nd = pd.DataFrame([{k: r[k] for k in ["title","publisher","published","sentiment","link"]} for r in sent_rows])
        st.dataframe(nd, use_container_width=True, hide_index=True)
    else:
        st.warning("No recent provider headlines were returned for this symbol.")

with tabs[12]:
    st.subheader("Alternative / forensic analysis")
    ac = st.columns(4)
    ac[0].metric("Piotroski F", f"{pio:.0f}/9" if np.isfinite(pio) else "N/A")
    ac[1].metric("Altman Z proxy", fmt(altz, 2))
    ac[2].metric("Beneish M", fmt(beneish, 2))
    ac[3].metric("DuPont ROE", f"{dup*100:.2f}%" if np.isfinite(dup) else "N/A")
    st.write("These models are screening tools with material industry and accounting-data limitations. A flag is not proof of distress, manipulation or misconduct.")
    if pio_notes:
        st.dataframe(pd.DataFrame({"Piotroski check": pio_notes}), use_container_width=True, hide_index=True)

with tabs[13]:
    st.subheader("SWOT generated from measurable signals")
    scols = st.columns(4)
    for col, (key, items) in zip(scols, swot.items()):
        with col:
            st.markdown(f"### {key}")
            for item in items or ["No strong signal from available data."]:
                st.write("• " + item)

with tabs[14]:
    st.subheader("ROE / fundamental company screener")
    st.caption("Example: set Minimum ROE % to 20 to find companies with ROE at or above 20%. The scan uses live/latest price data plus fetched financial statements. Scanning hundreds of companies can take time and may encounter provider rate limits.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current metadata pool", f"{len(pool):,}")
    c2.metric("ROE filter", f">= {min_roe:.0f}%")
    c3.metric("Max D/E", f"{max_de:.2f}")
    c4.metric("Min score", f"{min_score:.0f}")

    scan_size = st.selectbox("Deep scan size", [25, 50, 100, 250, 500, 1000], index=1)
    scan_size = min(scan_size, len(pool)) if len(pool) else 0
    sort_key = st.selectbox("Sort results by", ["Score","ROE %","Revenue Growth %","ROCE %","P/E","D/E","FCF (₹ cr)"])
    if st.button("🔎 Run deep screen", type="primary", disabled=scan_size == 0):
        params = {
            "min_pe": min_pe, "max_pe": max_pe, "min_roe": min_roe, "min_growth": min_growth,
            "max_de": max_de, "min_score": min_score, "min_roce": min_roce, "min_fcf": min_fcf,
            "require_data": require_data,
        }
        rows = pool.head(scan_size).copy()
        results: List[Dict[str, Any]] = []
        progress = st.progress(0.0)
        status = st.empty()
        workers = min(8, max(1, scan_size))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(scan_one, row, rf, market_return, params) for _, row in rows.iterrows()]
            for i, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                progress.progress(i / scan_size)
                status.caption(f"Scanned {i}/{scan_size}")
        status.empty()
        progress.empty()
        sdf = pd.DataFrame([r for r in results if r.get("_pass")])
        if not sdf.empty:
            sdf = sdf.drop(columns=["_pass"], errors="ignore")
            sdf = sdf.sort_values(sort_key, ascending=False, na_position="last")
            st.session_state["scan_result"] = sdf
        else:
            st.session_state["scan_result"] = pd.DataFrame()

    if "scan_result" in st.session_state:
        sdf = st.session_state["scan_result"]
        if sdf.empty:
            st.warning("No companies matched the current filters. Relax one or more filters or scan a larger pool.")
        else:
            st.success(f"Matched {len(sdf):,} companies")
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            st.download_button(
                "Download screener CSV",
                sdf.to_csv(index=False).encode("utf-8"),
                file_name="investorlens_screener.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "InvestorLens is an analytical research tool. Market data may be delayed, incomplete, rate-limited or revised. "
    "The optional live quote connector should be replaced with a licensed NSE/BSE or authorized real-time feed for exchange-grade real-time/commercial deployment."
)

with tabs[15]:
    st.subheader("Data Audit & Coverage")
    audit_rows = [
        ["Company", name],
        ["NSE symbol", sym],
        ["Statement provider", "Yahoo Finance / yfinance"],
        ["Quote provider", live.get("source", "Unavailable")],
        ["Statement currency", "INR (normalized to ₹ crore for display)"],
        ["Financial-year convention", "India: April–March"],
        ["Income statement periods", len(build_statement_history(fin, "income"))],
        ["Balance sheet periods", len(build_statement_history(fin, "balance"))],
        ["Cash flow periods", len(build_statement_history(fin, "cashflow"))],
        ["Last quote check", live.get("timestamp", "N/A")],
    ]
    st.dataframe(pd.DataFrame(audit_rows, columns=["Audit item","Value"]), use_container_width=True, hide_index=True)
    st.markdown("### 5-year coverage")
    coverage = pd.DataFrame({
        "Statement": ["Income Statement", "Balance Sheet", "Cash Flow"],
        "Periods available": [len(build_statement_history(fin,"income")), len(build_statement_history(fin,"balance")), len(build_statement_history(fin,"cashflow"))],
        "Required": [5,5,5],
    })
    coverage["Status"] = np.where(coverage["Periods available"] >= 5, "PASS", "INCOMPLETE")
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    st.warning("This build validates period coverage and prevents invented historical values. Provider data should still be reconciled against official NSE/company filings before public investment use.")
