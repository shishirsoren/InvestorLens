"""Market mover and industry comparison panels built on InvestorLens quote/history adapters.

This module intentionally displays only provider-returned prices/volumes. Missing index data is
omitted; historical-close fallbacks and daily-bar volume are labelled. Yahoo data is not promised
as exchange-grade realtime data. Auto-refresh uses Streamlit fragments (no polling while-loop).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    import streamlit as st
except ImportError:  # Pure return-calculation tests can run without deploying Streamlit.
    class _StreamlitTestStub:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(fn): return fn
            return decorator
    st = _StreamlitTestStub()

POLL_SECONDS = max(60, int(os.getenv("MARKET_POLL_SECONDS", "300")))
MAX_WORKERS = min(10, max(2, int(os.getenv("MARKET_DATA_WORKERS", "8"))))
PERIOD_SESSIONS = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}

# Only known NIFTY sectoral index symbols are attempted. An index appears only if its actual history is returned.
SECTOR_INDICES: Dict[str, Dict[str, Any]] = {
    "Nifty Bank": {"ticker": "^NSEBANK", "keywords": ["bank"]},
    "Nifty Financial Services": {"ticker": "^CNXFINANCE", "keywords": ["financial services", "finance", "bank", "insurance", "nbfc"]},
    "Nifty IT": {"ticker": "^CNXIT", "keywords": ["information technology", "software", "it services"]},
    "Nifty Auto": {"ticker": "^CNXAUTO", "keywords": ["automobile", "automotive", "auto components", "auto manufacturers"]},
    "Nifty FMCG": {"ticker": "^CNXFMCG", "keywords": ["fmcg", "consumer staples", "packaged foods", "household products"]},
    "Nifty Pharma": {"ticker": "^CNXPHARMA", "keywords": ["pharma", "drug manufacturers", "biotechnology"]},
    "Nifty Healthcare": {"ticker": "^CNXHEALTHCARE", "keywords": ["healthcare", "hospitals", "medical devices", "diagnostics"]},
    "Nifty Metal": {"ticker": "^CNXMETAL", "keywords": ["metal", "mining", "steel", "aluminium", "copper"]},
    "Nifty Realty": {"ticker": "^CNXREALTY", "keywords": ["real estate", "realty", "reit"]},
    "Nifty Energy": {"ticker": "^CNXENERGY", "keywords": ["energy", "oil gas", "power", "utilities"]},
    "Nifty PSU Bank": {"ticker": "^CNXPSUBANK", "keywords": ["public sector bank", "psu bank"]},
    "Nifty Media": {"ticker": "^CNXMEDIA", "keywords": ["media", "entertainment", "broadcasting"]},
    "Nifty Infrastructure": {"ticker": "^CNXINFRA", "keywords": ["infrastructure", "construction", "capital goods"]},
    "Nifty Cement": {"ticker": "^CNXCEMENT", "keywords": ["cement", "construction materials"]},
    "Nifty Consumer Durables": {"ticker": "^CNXCDUR", "keywords": ["consumer durables", "appliances"]},
    "Nifty Consumption": {"ticker": "^CNXCONSUM", "keywords": ["consumer services", "retail", "consumption"]},
}
BENCHMARK = "^CRSLDX"  # Yahoo Finance adapter's NIFTY 500 symbol, if available.


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v): return None
        x=float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _history(get_history: Callable, ticker: str, period: str="5y") -> pd.DataFrame:
    try:
        x=get_history(ticker,period)
        return x if isinstance(x,pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _quote(get_live_quote: Callable, ticker: str) -> Dict[str,Any]:
    try:
        x=get_live_quote(ticker)
        return x if isinstance(x,dict) else {}
    except Exception:
        return {}


def calculate_return(history: pd.DataFrame, period: str, latest_price: Optional[float]=None) -> Optional[float]:
    """Calculate period return from actual trading closes; never extrapolate missing history."""
    sessions=PERIOD_SESSIONS[period]
    if history is None or history.empty or "Close" not in history.columns: return None
    closes=pd.to_numeric(history["Close"],errors="coerce").dropna()
    if len(closes)<sessions+1: return None
    base=_num(closes.iloc[-(sessions+1)])
    end=_num(latest_price)
    if end is None: end=_num(closes.iloc[-1])
    if base is None or end is None or base<=0: return None
    return (end/base-1)*100


def _symbol_volume(q: Mapping[str,Any], history: pd.DataFrame) -> Optional[float]:
    direct=_num(q.get("volume"))
    if direct is not None: return direct
    if history is not None and not history.empty and "Volume" in history.columns:
        s=pd.to_numeric(history["Volume"],errors="coerce").dropna()
        return _num(s.iloc[-1]) if len(s) else None
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _scan_movers(symbols: Tuple[Tuple[str,str,str],...], _get_live_quote: Callable, _get_history: Callable) -> pd.DataFrame:
    rows=[]
    def fetch(item):
        sym,company,industry=item
        q=_quote(_get_live_quote,sym)
        h=pd.DataFrame()
        direct_volume=_num(q.get("volume"))
        volume=direct_volume
        volume_source="Provider quote volume" if direct_volume is not None else "Unavailable"
        if volume is None:
            h=_history(_get_history,sym,"5d")
            volume=_symbol_volume(q,h)
            volume_source="Latest daily-bar volume" if volume is not None else "Unavailable"
        price=_num(q.get("price")); prev=_num(q.get("prev_close")); pct=_num(q.get("change_pct"))
        if h.empty and (price is None or pct is None): h=_history(_get_history,sym,"5d")
        if h is not None and not h.empty and "Close" in h.columns:
            c=pd.to_numeric(h["Close"],errors="coerce").dropna()
            if len(c)>=2:
                if price is None: price=_num(c.iloc[-1])
                if prev is None: prev=_num(c.iloc[-2])
        if pct is None and price is not None and prev not in (None,0): pct=(price/prev-1)*100
        if price is None or pct is None: return None
        return {"Symbol":sym,"Company":company or sym,"Industry":industry or "Unclassified","Price (₹)":price,
                "Change %":pct,"Volume":volume,"Volume source":volume_source,"Previous close (₹)":prev,
                "Quote source":q.get("source","Latest historical close (provider quote unavailable)"),
                "Quote checked":q.get("timestamp","")}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs=[ex.submit(fetch,x) for x in symbols]
        for f in as_completed(futs):
            try:
                r=f.result()
                if r is not None: rows.append(r)
            except Exception: pass
    if not rows:
        return pd.DataFrame(columns=["Symbol","Company","Industry","Price (₹)","Change %","Volume","Volume source","Previous close (₹)","Quote source","Quote checked"])
    return pd.DataFrame(rows).sort_values("Change %",ascending=False).reset_index(drop=True)


def _sector_performance(get_history: Callable,get_live_quote: Callable,period: str) -> pd.DataFrame:
    rows=[]
    def fetch(name,meta):
        ticker=meta["ticker"]
        h=_history(get_history,ticker,"5y")
        if h.empty or "Close" not in h.columns:return None
        q=_quote(get_live_quote,ticker); current=_num(q.get("price"))
        c=pd.to_numeric(h["Close"],errors="coerce").dropna()
        if current is None and len(c): current=_num(c.iloc[-1])
        r=calculate_return(h,period,current)
        if r is None:return None
        prev=_num(c.iloc[-2]) if len(c)>1 else None
        one_day=(current/prev-1)*100 if current is not None and prev not in (None,0) else None
        quote_source=q.get("source","Latest historical close (quote unavailable)") if _num(q.get("price")) is not None else "Latest historical close (quote unavailable)"
        return {"Industry":name,"Index symbol":ticker,"Latest index value":current,"1D %":one_day,"Period return %":r,
                "Quote source":quote_source,"History rows":len(c)}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS,8)) as ex:
        fs=[ex.submit(fetch,n,m) for n,m in SECTOR_INDICES.items()]
        for f in as_completed(fs):
            try:
                row=f.result()
                if row:rows.append(row)
            except Exception:pass
    return pd.DataFrame(rows).sort_values("Period return %",ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _industry_members(universe: pd.DataFrame, industry: str) -> pd.DataFrame:
    if universe is None or universe.empty or "symbol" not in universe.columns: return pd.DataFrame()
    keywords=SECTOR_INDICES[industry]["keywords"]
    sector=universe.get("sector",pd.Series("",index=universe.index)).fillna("").astype(str).str.lower()
    mask=sector.apply(lambda s:any(k in s for k in keywords))
    return universe.loc[mask].drop_duplicates("symbol").copy()


@st.cache_data(ttl=3600, show_spinner=False)
def _peer_ratios(symbols: Tuple[Tuple[str,str],...], _get_info: Callable, _get_financials: Callable, _calc_ratios: Callable) -> pd.DataFrame:
    rows=[]
    def fetch(item):
        sym,sector=item
        try:
            info=_get_info(sym) or {}; fin=_get_financials(sym) or {}; r=_calc_ratios(sym,fin,info,np.nan)
            return {"Symbol":sym,"Company":info.get("longName") or sym,"Industry":info.get("industry") or sector,
                    "Revenue Growth %":_num(r.get("Revenue Growth %")),"Net Profit Growth %":_num(r.get("Net Profit Growth %")),
                    "Net Margin %":_num(r.get("Net Margin %")),"ROE %":_num(r.get("ROE %")),"ROCE %":_num(r.get("ROCE %")),
                    "P/E":_num(r.get("PE")),"Debt / Equity":_num(r.get("Debt / Equity"))}
        except Exception:return None
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS,8)) as ex:
        fs=[ex.submit(fetch,x) for x in symbols]
        for f in as_completed(fs):
            try:
                r=f.result()
                if r:rows.append(r)
            except Exception:pass
    return pd.DataFrame(rows)


def _favorable(metric: str, delta: Optional[float], financial: bool) -> str:
    if delta is None or not np.isfinite(delta):return "N/A — insufficient data"
    if metric=="P/E":return "Below peer P/E (valuation discount; check quality)" if delta<0 else "Above peer P/E (premium needs justification)" if delta>0 else "At peer mean"
    if metric=="Debt / Equity" and financial:return "Sector-dependent; not scored as ordinary corporate leverage"
    if delta==0:return "At peer mean"
    higher_good=metric in ["Revenue Growth %","Net Profit Growth %","Net Margin %","ROE %","ROCE %"]
    favourable=(delta>0) if higher_good else (delta<0)
    return "Above peer mean — generally favourable" if favourable and delta>0 else "Below peer mean — generally favourable" if favourable else "Above peer mean — review risk/quality" if delta>0 else "Below peer mean — review growth/quality"


def render_market_industry_panel(universe: pd.DataFrame, get_history: Callable, get_live_quote: Callable,
                                get_info: Callable, get_financials: Callable, calc_ratios: Callable,
                                selected_symbol: str="", selected_ratios: Optional[Mapping[str,Any]]=None,
                                selected_sector: str="") -> None:
    st.header("🌐 Live Market & Industry Intelligence")
    st.caption(f"Market movers automatically re-poll about every {POLL_SECONDS} seconds. Sources are the existing InvestorLens quote/history adapters; missing data is not fabricated.")
    period=st.segmented_control("Industry return period",options=list(PERIOD_SESSIONS.keys()),default="1D",key="il_sector_period")
    if period not in PERIOD_SESSIONS:period="1D"

    @st.fragment(run_every=POLL_SECONDS)
    def live_market_charts():
        st.subheader("Indian market movers")
        if universe is None or universe.empty:
            st.warning("No stock universe loaded. Refresh the NIFTY Total Market universe first.")
        else:
            symbols=[]
            for _,row in universe.iterrows():
                sym=str(row.get("symbol","")).strip().upper()
                if sym and sym!="NAN": symbols.append((sym,str(row.get("company",sym)),str(row.get("sector",""))))
            st.caption(f"Attempting provider-backed latest prices for {len(symbols):,} companies. The display may contain fewer rows if quotes are unavailable/rate-limited.")
            with st.spinner("Loading provider quotes and latest available volume…"):
                snap=_scan_movers(tuple(symbols),get_live_quote,get_history)
            if snap.empty:
                st.warning("The configured provider did not return usable quote/change data. No prices have been fabricated.")
            else:
                gain=snap.sort_values("Change %",ascending=False).head(10)
                loss=snap.sort_values("Change %",ascending=True).head(10)
                active=snap.dropna(subset=["Volume"]).sort_values("Volume",ascending=False).head(10)
                c1,c2,c3=st.columns(3)
                with c1:
                    st.markdown("### 🟢 Top gainers")
                    st.dataframe(gain[["Symbol","Company","Price (₹)","Change %"]].round(2),use_container_width=True,hide_index=True)
                with c2:
                    st.markdown("### 🔴 Top losers")
                    st.dataframe(loss[["Symbol","Company","Price (₹)","Change %"]].round(2),use_container_width=True,hide_index=True)
                with c3:
                    st.markdown("### 🔊 Most active")
                    if active.empty: st.info("No volume field was returned by the provider.")
                    else: st.dataframe(active[["Symbol","Company","Volume","Volume source","Change %"]].round(2),use_container_width=True,hide_index=True)
                with st.expander("Full market snapshot / data source"):
                    st.dataframe(snap,use_container_width=True,hide_index=True)
                    st.download_button("Download market snapshot CSV",snap.to_csv(index=False).encode("utf-8"),"investorlens_market_snapshot.csv","text/csv",key="il_download_market")
                st.caption(f"Usable snapshot: {len(snap):,} of {len(symbols):,} symbols. Volume source is shown per row; daily-bar volume is not intraday cumulative volume.")

        st.subheader(f"Live Industry Performance · {period}")
        industries=_sector_performance(get_history,get_live_quote,period)
        if industries.empty:
            st.warning("No supported sector-index histories were returned by the provider. Indices without history are omitted.")
        else:
            bench_h=_history(get_history,BENCHMARK,"5y"); bench_q=_quote(get_live_quote,BENCHMARK)
            bench_r=calculate_return(bench_h,period,_num(bench_q.get("price")))
            industries["vs NIFTY 500 (pp)"]=industries["Period return %"].apply(lambda x:x-bench_r if bench_r is not None else np.nan)
            top=industries.sort_values("Period return %",ascending=False).head(8)
            bottom=industries.sort_values("Period return %",ascending=True).head(8)
            a,b=st.columns(2)
            with a:
                st.markdown("### 🟢 Top Performing Industries")
                fig=go.Figure(go.Bar(x=top["Period return %"],y=top["Industry"],orientation="h",marker_color="#16a34a",text=top["Period return %"].map(lambda x:f"{x:.2f}%"),textposition="outside"))
                fig.update_layout(height=390,margin=dict(l=5,r=35,t=10,b=5),xaxis_title=f"Return ({period}) %",yaxis_title="",yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
            with b:
                st.markdown("### 🔴 Underperforming Industries")
                fig=go.Figure(go.Bar(x=bottom["Period return %"],y=bottom["Industry"],orientation="h",marker_color="#dc2626",text=bottom["Period return %"].map(lambda x:f"{x:.2f}%"),textposition="outside"))
                fig.update_layout(height=390,margin=dict(l=5,r=35,t=10,b=5),xaxis_title=f"Return ({period}) %",yaxis_title="",yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
            st.dataframe(industries.round(3),use_container_width=True,hide_index=True)
            st.caption(f"The index universe is compared with NIFTY 500 ({BENCHMARK}) only if the provider supplies sufficient history. Unsupported indices are not filled with synthetic returns.")

    live_market_charts()

    st.divider()
    st.subheader("Industry classification, peer averages & analytical recommendation")
    industry=st.selectbox("Industry index for peer analysis",list(SECTOR_INDICES.keys()),index=0,key="il_peer_sector")
    members=_industry_members(universe,industry)
    if members.empty:
        st.info("No constituents have a matching industry label in the loaded universe. The actual index performance above may still be available; no stocks are force-assigned to this peer group.")
        return
    st.caption(f"{len(members):,} companies in the loaded universe match the constituent industry labels for this peer group. Financial averages are calculated only from companies returning usable fundamentals.")
    if "market_cap_cr" in members and members["market_cap_cr"].notna().any(): members=members.sort_values("market_cap_cr",ascending=False,na_position="last")
    coverage=st.selectbox("Peer scan scope",[30,60,100,"All matching companies"],index=0,key="il_peer_scope")
    sample_n=len(members) if coverage=="All matching companies" else min(int(coverage),len(members))
    if st.button("Calculate / refresh industry averages",key="il_peer_run"):
        peer_symbols=tuple((str(r["symbol"]),str(r.get("sector",""))) for _,r in members.head(sample_n).iterrows())
        with st.spinner(f"Fetching statement data for up to {sample_n} companies…"):
            peers=_peer_ratios(peer_symbols,get_info,get_financials,calc_ratios)
        st.session_state["il_peer_data"]={"industry":industry,"members":len(members),"requested":sample_n,"data":peers}
    pdata=st.session_state.get("il_peer_data",{})
    if pdata.get("industry")==industry and isinstance(pdata.get("data"),pd.DataFrame):
        peers=pdata["data"]
        st.caption(f"Peer data returned for {len(peers):,} of {sample_n:,} requested companies out of {len(members):,} matched. Means below are a partial sample unless coverage is complete.")
        metrics=["Revenue Growth %","Net Profit Growth %","Net Margin %","ROE %","ROCE %","P/E","Debt / Equity"]
        avg=[]
        for metric in metrics:
            values=pd.to_numeric(peers[metric],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna() if metric in peers else pd.Series(dtype=float)
            avg.append({"Metric":metric,"Industry sample mean":values.mean() if len(values) else np.nan,"Industry sample median":values.median() if len(values) else np.nan,"Companies with metric":len(values)})
        avgdf=pd.DataFrame(avg)
        st.dataframe(avgdf.round(3),use_container_width=True,hide_index=True)
        if selected_symbol and selected_ratios:
            st.markdown(f"### {selected_symbol} vs {industry} sample average")
            financial=any(k in (selected_sector or "").lower() for k in ["bank","financial","insurance","nbfc"])
            comp=[]
            for metric,rkey in [("Revenue Growth %","Revenue Growth %"),("Net Profit Growth %","Net Profit Growth %"),("Net Margin %","Net Margin %"),("ROE %","ROE %"),("ROCE %","ROCE %"),("P/E","PE"),("Debt / Equity","Debt / Equity")]:
                own=_num(selected_ratios.get(rkey)); peer=_num(avgdf.loc[avgdf["Metric"]==metric,"Industry sample mean"].iloc[0]); delta=own-peer if own is not None and peer is not None else None
                comp.append({"Metric":metric,"Selected company":own,"Peer mean":peer,"Difference":delta,"Reading":_favourable(metric,delta,financial)})
            st.dataframe(pd.DataFrame(comp).round(3),use_container_width=True,hide_index=True)
        meta=SECTOR_INDICES[industry]; ih=_history(get_history,meta["ticker"],"5y"); iq=_quote(get_live_quote,meta["ticker"]); ir=calculate_return(ih,period,_num(iq.get("price")))
        bh=_history(get_history,BENCHMARK,"5y"); bq=_quote(get_live_quote,BENCHMARK); br=calculate_return(bh,period,_num(bq.get("price")))
        def peer_mean(col):
            s=pd.to_numeric(peers[col],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna() if col in peers else pd.Series(dtype=float)
            return float(s.mean()) if len(s) else None
        revg,netg,roe=peer_mean("Revenue Growth %"),peer_mean("Net Profit Growth %"),peer_mean("ROE %")
        excess=ir-br if ir is not None and br is not None else None
        reasons=[]; risks=[]; signal="Neutral"
        if excess is not None: reasons.append(f"{industry} returned {ir:.2f}% over {period}; NIFTY 500 returned {br:.2f}%; relative return {excess:+.2f} percentage points.")
        else: risks.append("Industry index or NIFTY 500 history is unavailable for this period.")
        if revg is not None: reasons.append(f"Peer-sample mean revenue growth: {revg:.2f}%.")
        else: risks.append("Revenue growth field is unavailable for the peer sample.")
        if netg is not None: reasons.append(f"Peer-sample mean net-profit growth: {netg:.2f}%.")
        else: risks.append("Net-profit growth field is unavailable for the peer sample.")
        if roe is not None: reasons.append(f"Peer-sample mean ROE: {roe:.2f}%.")
        else: risks.append("ROE is unavailable for the peer sample.")
        if excess is not None and None not in [revg,netg,roe]:
            if excess>0 and revg>0 and netg>0 and roe>0: signal="Positive"
            elif excess<0 and (revg<0 or netg<0 or roe<0): signal="Caution"
        st.markdown(f"### Industry signal: **{signal}**")
        for x in reasons: st.write("• "+x)
        for x in risks: st.warning(x)
        st.caption("Signal is a transparent rules-based interpretation of actual index returns and retrieved peer fundamentals. It is not a forecast or personalized investment advice. Industry sample coverage is shown above.")
    st.caption("Current data source is InvestorLens' existing market quote/history and financial-statement adapter. Yahoo Finance data is not guaranteed to be exchange-grade real-time data; provider-approved WebSockets require a compatible licensed feed.")
