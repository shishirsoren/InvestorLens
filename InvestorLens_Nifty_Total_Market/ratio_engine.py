"""InvestorLens supplemental ratio engine with period-aware calculations and audit metadata.

All arithmetic uses provider statement units without scaling. Monetary rows are converted to
INR crore only for display; per-share values remain INR/share. Missing values are never imputed.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math, re
import numpy as np
import pandas as pd

NA = "N/A — insufficient data"

# Existing ratio matrix (app.py calc_ratios) — preserve, do not duplicate in the supplemental table.
EXISTING = {
    "Revenue Growth %", "Net Profit Growth %", "Gross Margin %", "Operating Margin %",
    "Net Margin %", "ROE %", "ROA %", "ROCE %", "Current Ratio", "Quick Ratio",
    "Debt / Equity", "Debt / Assets %", "Net Debt (₹ cr)", "Interest Coverage",
    "Debt / EBITDA", "Asset Turnover", "Equity Multiplier", "CFO (₹ cr)",
    "Capex (₹ cr)", "FCF (₹ cr)", "Cash Conversion", "PE", "Forward PE", "PB",
    "EV/EBITDA", "PEG", "Dividend Yield %", "Payout Ratio %", "Market Cap (₹ cr)", "Beta (vendor)"
}

# Each field is mapped to statement type and row aliases common to yfinance statement objects.
FIELDS: Dict[str, Tuple[str, List[str], str]] = {
    "revenue": ("income", ["Total Revenue", "Operating Revenue", "Revenue"], "Revenue"),
    "gross_profit": ("income", ["Gross Profit"], "Gross Profit"),
    "ebitda": ("income", ["EBITDA", "Normalized EBITDA"], "EBITDA"),
    "ebit": ("income", ["EBIT", "Operating Income"], "EBIT"),
    "operating_income": ("income", ["Operating Income", "EBIT"], "Operating Income"),
    "net_income": ("income", ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"], "Net Income"),
    "pretax": ("income", ["Pretax Income", "Pre Tax Income", "Income Before Tax"], "Pretax Income"),
    "tax": ("income", ["Tax Provision", "Income Tax Expense"], "Tax Provision"),
    "cogs": ("income", ["Cost Of Revenue", "Cost of Goods Sold", "Cost Of Goods Sold"], "Cost of Revenue / COGS"),
    "total_expenses": ("income", ["Total Expenses", "Total Operating Expenses"], "Total Expenses"),
    "interest_expense": ("income", ["Interest Expense", "Interest Expense Non Operating"], "Interest Expense"),
    "eps": ("income", ["Diluted EPS", "Basic EPS", "Diluted EPS Continuing Operations"], "Reported EPS"),
    "current_assets": ("balance", ["Current Assets"], "Current Assets"),
    "current_liabilities": ("balance", ["Current Liabilities"], "Current Liabilities"),
    "cash": ("balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"], "Cash and Cash Equivalents"),
    "short_investments": ("balance", ["Other Short Term Investments", "Short Term Investments", "Marketable Securities"], "Short-term Investments"),
    "receivables": ("balance", ["Accounts Receivable", "Receivables", "Net Receivables"], "Trade Receivables"),
    "inventory": ("balance", ["Inventory", "Inventory Net", "Inventories"], "Inventory"),
    "payables": ("balance", ["Accounts Payable", "Trade Payables", "Payables And Accrued Expenses"], "Trade Payables"),
    "assets": ("balance", ["Total Assets"], "Total Assets"),
    "equity": ("balance", ["Common Stock Equity", "Stockholders Equity", "Stockholders' Equity", "Total Stockholder Equity"], "Shareholders' Equity"),
    "liabilities": ("balance", ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], "Total Liabilities"),
    "debt": ("balance", ["Total Debt"], "Total Debt"),
    "long_debt": ("balance", ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], "Long-term Debt"),
    "ppe": ("balance", ["Net PPE", "Property Plant Equipment", "Net Property Plant And Equipment"], "Net PPE"),
    "goodwill": ("balance", ["Goodwill"], "Goodwill"),
    "intangibles": ("balance", ["Other Intangible Assets", "Intangible Assets Net Of Goodwill", "Finite Lived Intangible Assets Net"], "Other Intangible Assets"),
    "retained": ("balance", ["Retained Earnings", "Retained Earnings And Accumulated Deficit"], "Retained Earnings"),
    "shares": ("balance", ["Ordinary Shares Number", "Share Issued", "Common Shares Outstanding"], "Shares Outstanding"),
    "preferred": ("balance", ["Preferred Stock Equity", "Preferred Stock"], "Preferred Equity"),
    "nci": ("balance", ["Minority Interest", "Noncontrolling Interest In Consolidated Entity"], "Non-controlling Interests"),
    "cfo": ("cashflow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"], "Cash Flow from Operations"),
    "capex": ("cashflow", ["Capital Expenditure", "Capital Expenditure Reported"], "Capital Expenditure"),
    "depreciation": ("cashflow", ["Depreciation And Amortization", "Depreciation"], "Depreciation & Amortization"),
    "dividends_paid": ("cashflow", ["Cash Dividends Paid", "Common Stock Dividend Paid", "Dividends Paid"], "Dividends Paid"),
    "buybacks": ("cashflow", ["Repurchase Of Capital Stock", "Common Stock Payments"], "Share Buybacks"),
    "share_issuance": ("cashflow", ["Issuance Of Capital Stock", "Common Stock Issuance"], "Share Issuance"),
    # Optional disclosure fields frequently absent from summary financial APIs.
    "credit_sales": ("any", ["Credit Sales"], "Credit Sales"),
    "credit_purchases": ("any", ["Credit Purchases"], "Credit Purchases"),
    "net_interest_income": ("any", ["Net Interest Income", "Net Interest Revenue"], "Net Interest Income"),
    "earning_assets": ("any", ["Average Earning Assets", "Average Interest Earning Assets"], "Average Earning Assets"),
    "casa": ("any", ["CASA Deposits", "Current And Savings Account Deposits"], "CASA Deposits"),
    "deposits": ("any", ["Total Deposits"], "Total Deposits"),
    "gross_npa": ("any", ["Gross NPA", "Gross Non Performing Assets"], "Gross NPA"),
    "gross_advances": ("any", ["Gross Advances"], "Gross Advances"),
    "net_npa": ("any", ["Net NPA", "Net Non Performing Assets"], "Net NPA"),
    "net_advances": ("any", ["Net Advances"], "Net Advances"),
    "npa_provisions": ("any", ["NPA Provisions", "Provisions For NPAs"], "NPA Provisions"),
    "reg_capital": ("any", ["Eligible Regulatory Capital", "Regulatory Capital"], "Eligible Regulatory Capital"),
    "rwa": ("any", ["Risk Weighted Assets"], "Risk Weighted Assets"),
    "solvency_available": ("any", ["Available Solvency Margin"], "Available Solvency Margin"),
    "solvency_required": ("any", ["Required Solvency Margin"], "Required Solvency Margin"),
    "claims": ("any", ["Claims Incurred"], "Claims Incurred"),
    "earned_premiums": ("any", ["Earned Premiums", "Net Earned Premium"], "Earned Premiums"),
    "insurance_expenses": ("any", ["Insurance Operating Expenses", "Relevant Expenses"], "Insurance Expenses"),
    "rd": ("income", ["Research And Development", "Research Development Expense", "Research And Development Expense"], "R&D Expense"),
    "beds": ("any", ["Average Operational Beds", "Operational Beds", "Beds In Operation"], "Average Operational Beds"),
    "employees": ("any", ["Average Employee Count", "Average Number Of Employees", "Full Time Employees", "Number Of Employees", "Employees"], "Average Employee Count"),
    "billable_hours": ("any", ["Billable Employee Hours"], "Billable Employee Hours"),
    "available_hours": ("any", ["Available Employee Hours"], "Available Employee Hours"),
    "telecom_revenue": ("any", ["Telecom Service Revenue", "Relevant Service Revenue"], "Telecom Service Revenue"),
    "subscribers": ("any", ["Average Subscribers", "Average Subscribers Count"], "Average Subscribers"),
    "units_sold": ("any", ["Units Sold", "Sales Volume Units"], "Units Sold"),
    "cash_available_debt_service": ("any", ["Cash Available For Debt Service", "Cash Flow Available For Debt Service"], "Cash Available for Debt Service"),
    "principal_repayments": ("cashflow", ["Repayment Of Debt", "Repayments Of Long Term Debt"], "Principal Repayments"),
    "interest_paid": ("cashflow", ["Interest Paid", "Interest Paid Supplemental Data"], "Interest Paid"),
}


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v): return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _cols(df: Any) -> List[Any]:
    if not isinstance(df, pd.DataFrame) or df.empty: return []
    out=[]
    for c in df.columns:
        try: out.append((pd.to_datetime(c), c))
        except Exception: pass
    return [c for _,c in sorted(out, key=lambda x:x[0])]


def _fy_end_year(date: Any) -> int:
    d=pd.to_datetime(date)
    return int(d.year + 1 if d.month >= 4 else d.year)


def _fy_label(y: Optional[int]) -> str:
    return f"FY{y-1}-{str(y)[-2:]}" if y else "No matching FY"


def _find_row(df: Any, aliases: Sequence[str]) -> Optional[Any]:
    if not isinstance(df, pd.DataFrame) or df.empty: return None
    rows={_norm(i):i for i in df.index}
    for a in aliases:
        if _norm(a) in rows: return rows[_norm(a)]
    for a in aliases:
        n=_norm(a)
        for k, original in rows.items():
            # avoid broad/ambiguous short substring matching
            if len(n) >= 7 and (n in k or k in n):
                if n == "minorityinterest" and k != n: continue
                return original
    return None


def _extract_series(df: Any, aliases: Sequence[str]) -> pd.Series:
    row=_find_row(df, aliases)
    if row is None: return pd.Series(dtype=float)
    data={}
    for col in _cols(df):
        y=_fy_end_year(col); v=_num(df.loc[row,col])
        if v is not None: data[y]=v # latest date for each same FY wins
    return pd.Series(data, dtype=float).sort_index()


def _metric_series(fin: Mapping[str, Any], metric: str) -> pd.Series:
    kind, aliases, _ = FIELDS[metric]
    kinds=["income","balance","cashflow"] if kind=="any" else [kind]
    result=pd.Series(dtype=float)
    for k in kinds:
        annual_names={"income":["income"],"balance":["balance"],"cashflow":["cashflow"]}[k]
        q_names={"income":["q_income"],"balance":["q_balance"],"cashflow":["q_cashflow"]}[k]
        annual=pd.Series(dtype=float)
        for name in annual_names:
            df=fin.get(name)
            if isinstance(df,pd.DataFrame) and not df.empty:
                annual=_extract_series(df,aliases)
                if not annual.empty: break
        qser=pd.Series(dtype=float)
        for name in q_names:
            df=fin.get(name)
            if not isinstance(df,pd.DataFrame) or df.empty: continue
            row=_find_row(df,aliases)
            cols=_cols(df)
            if row is None or not cols: continue
            groups: Dict[int,List[Tuple[pd.Timestamp,float]]]={}
            for col in cols:
                val=_num(df.loc[row,col])
                if val is not None: groups.setdefault(_fy_end_year(col),[]).append((pd.to_datetime(col),val))
            vals={}
            for year, arr in groups.items():
                arr.sort(key=lambda z:z[0])
                if k=="balance":
                    march=[v for d,v in arr if d.month==3]
                    if march: vals[year]=march[-1]
                elif len(arr)>=4:
                    # Only aggregate full Apr-Mar four-quarter flows.
                    vals[year]=float(sum(v for _,v in arr[-4:]))
            qser=pd.Series(vals,dtype=float).sort_index()
            break
        local=annual.copy()
        for y,v in qser.items():
            if y not in local.index: local.loc[y]=v
        if local.empty: local=qser
        if not local.empty:
            for y,v in local.items():
                if y not in result.index: result.loc[y]=v
    return result.sort_index()


def _at(fin: Mapping[str,Any], metric: str, year: Optional[int]) -> Optional[float]:
    if year is None: return None
    s=_metric_series(fin,metric)
    return _num(s.get(year)) if year in s.index else None


def _latest_year(fin: Mapping[str,Any], fields: Sequence[str]) -> Optional[int]:
    ss=[_metric_series(fin,k) for k in fields]
    available=[s for s in ss if not s.empty]
    if not available: return None
    common=set(int(y) for y in available[0].index)
    for s in available[1:]: common &= set(int(y) for y in s.index)
    return max(common) if common else int(max(available[0].index))


def _safe(n: Optional[float], d: Optional[float], mult: float=1.0, allow_negative_d: bool=False) -> Tuple[Optional[float],str]:
    if n is None or d is None: return None,"required input missing"
    if not math.isfinite(n) or not math.isfinite(d): return None,"input is null or non-finite"
    if d==0: return None,"denominator is zero"
    if d<0 and not allow_negative_d: return None,"denominator is negative; ratio not interpreted"
    x=n/d*mult
    return (x,"") if math.isfinite(x) else (None,"result is non-finite")


def calculate_peg(fin: Mapping[str,Any], info: Mapping[str,Any]) -> Tuple[Optional[float],str]:
    """PEG = trailing P/E / annual EPS growth percentage points (20% growth -> 20)."""
    pe=_num(info.get("trailingPE")); eps=_metric_series(fin,"eps")
    if pe is None or pe<=0: return None,"trailing P/E missing, zero or negative"
    if eps.empty: return None,"reported annual EPS history unavailable"
    y=int(max(eps.index)); cur=_num(eps.get(y)); prev=_num(eps.get(y-1))
    if cur is None or prev is None: return None,"current/prior annual EPS missing"
    if prev==0: return None,"prior-year EPS is zero"
    g=(cur-prev)/abs(prev)*100
    if g<=0: return None,"EPS growth is zero or negative; PEG not meaningful"
    return pe/g,""


def calculate_piotroski(fin: Mapping[str,Any]) -> Tuple[float, List[str]]:
    """Return full nine-test Piotroski only when each test's inputs are present."""
    keys=["net_income","cfo","assets","long_debt","current_assets","current_liabilities","revenue","gross_profit","shares"]
    y=_latest_year(fin,keys)
    if y is None: return np.nan,["No common annual period across required statements"]
    cur={k:_at(fin,k,y) for k in keys}; prev={k:_at(fin,k,y-1) for k in keys}; assets2=_at(fin,"assets",y-2)
    missing=[]
    for k in keys:
        if cur[k] is None: missing.append(FIELDS[k][2]+" current FY")
        if prev[k] is None: missing.append(FIELDS[k][2]+" prior FY")
    if assets2 is None: missing.append("Total Assets two years prior (for prior ROA average)")
    if missing: return np.nan,missing
    if cur["assets"]<=0 or prev["assets"]<=0 or assets2<=0 or cur["current_liabilities"]<=0 or prev["current_liabilities"]<=0 or cur["revenue"]<=0 or prev["revenue"]<=0:
        return np.nan,["Positive assets, current liabilities and revenue required for all nine tests"]
    avg_assets=(cur["assets"]+prev["assets"])/2
    avg_assets_prev=(prev["assets"]+assets2)/2
    roa=cur["net_income"]/avg_assets; roa_prev=prev["net_income"]/avg_assets_prev
    checks=[
        roa>0,
        cur["cfo"]>0,
        roa>roa_prev,
        cur["cfo"]>cur["net_income"],
        cur["long_debt"]/cur["assets"] < prev["long_debt"]/prev["assets"],
        cur["current_assets"]/cur["current_liabilities"] > prev["current_assets"]/prev["current_liabilities"],
        cur["shares"]<=prev["shares"],
        cur["gross_profit"]/cur["revenue"] > prev["gross_profit"]/prev["revenue"],
        cur["revenue"]/avg_assets > prev["revenue"]/avg_assets_prev,
    ]
    labels=["Positive ROA","Positive operating cash flow","ROA improved","CFO exceeds net income","Long-term leverage declined","Current ratio improved","No share dilution (shares-issued proxy)","Gross margin improved","Asset turnover improved"]
    return float(sum(bool(x) for x in checks)),[f"{label}: {'pass' if ok else 'fail'}" for label,ok in zip(labels,checks)]


def calculate_altman_z(fin: Mapping[str,Any], info: Optional[Mapping[str,Any]]=None, sector: str="") -> Tuple[float,str,Optional[int],List[str],str]:
    """Calculate an appropriate published Z-score variant only when all period-aligned inputs exist."""
    info=dict(info or {}); industry=str(info.get("industry") or sector or "").lower()
    financial=any(k in industry for k in ["bank","insurance","financial service","financial institution","nbfc"])
    manufacturing=any(k in industry for k in ["manufacturing","auto manufacturers","steel","aluminum","chemicals","pharmaceuticals","electrical equipment","building materials","industrial machinery","tyre manufacturers","copper"])
    nonmanufacturing=any(k in industry for k in ["software","information technology","it services","telecom","healthcare providers","hospital","retail","consumer services","transportation","utilities","media","professional services","real estate operations","oil and gas","energy services"])
    y=_latest_year(fin,["current_assets","current_liabilities","assets","retained","ebit","liabilities","revenue"])
    vals={k:_at(fin,k,y) for k in ["current_assets","current_liabilities","assets","retained","ebit","liabilities","revenue"]} if y else {}
    mve=_num(info.get("marketCapAtFiscalYearEnd"))
    missing=[FIELDS[k][2] for k in vals if vals[k] is None]
    if mve is None: missing.append("Market value of equity at matching FY-end")
    if financial: return np.nan,"Not applicable — financial-company-specific model required",y,missing,"Ordinary-company Altman variants are not suitable for banks, NBFCs or insurers."
    if not (manufacturing or nonmanufacturing):
        missing.append("Verified manufacturing/non-manufacturing company classification")
        return np.nan,"Company type not established",y,missing,"A variant is not selected from an ambiguous industry label."
    assets=vals.get("assets"); liab=vals.get("liabilities")
    if missing or assets in (None,0) or liab in (None,0):
        return np.nan,"Required inputs unavailable",y,missing or ["Total assets/liabilities must be nonzero"],"Current market capitalization is not substituted for FY-end market equity."
    wc=vals["current_assets"]-vals["current_liabilities"]
    if manufacturing:
        formula="1.2×(Working Capital/Assets) + 1.4×(Retained Earnings/Assets) + 3.3×(EBIT/Assets) + 0.6×(Market Equity/Total Liabilities) + Revenue/Assets"
        score=1.2*wc/assets+1.4*vals["retained"]/assets+3.3*vals["ebit"]/assets+0.6*mve/liab+vals["revenue"]/assets
    else:
        formula="3.25 + 6.56×(Working Capital/Assets) + 3.26×(Retained Earnings/Assets) + 6.72×(EBIT/Assets) + 1.05×(Market Equity/Total Liabilities)"
        score=3.25+6.56*wc/assets+3.26*vals["retained"]/assets+6.72*vals["ebit"]/assets+1.05*mve/liab
    return float(score),formula,y,[],"Use only with the matching published model and its corresponding interpretation thresholds."

def calculate_ratio_audit(financials: Mapping[str,Any], info: Optional[Mapping[str,Any]]=None,
                          price: Optional[float]=None, dividends: Optional[pd.Series]=None,
                          sector: str="", current_ratios: Optional[Mapping[str,Any]]=None) -> pd.DataFrame:
    fin=dict(financials or {}); info=dict(info or {}); px=_num(price) or _num(info.get("currentPrice"))
    sector_text=(str(sector or "")+" "+str(info.get("industry") or "")).lower()
    results=[]
    def put(category,name,formula,value,unit,year,fields,why,missing="",limitation="",reason=""):
        if name in EXISTING and not (name=="PEG"): return
        if limitation.startswith("Not applicable"):
            status="Not applicable — sector"
        elif value is None or not math.isfinite(value):
            status=NA
            if reason and not (reason == "required input missing" and missing): missing=(missing+"; " if missing else "")+reason
        else: status="Calculated"
        if value is None or not math.isfinite(value): display=NA
        elif unit=="%": display=f"{value:,.2f}%"
        elif unit=="x": display=f"{value:,.2f}x"
        elif unit=="days": display=f"{value:,.1f} days"
        elif unit=="₹ crore": display=f"₹{value/1e7:,.2f} crore"
        elif unit=="₹/share": display=f"₹{value:,.2f}"
        elif unit=="₹/employee": display=f"₹{value:,.2f}"
        elif unit=="₹/bed": display=f"₹{value:,.2f}"
        elif unit=="₹/subscriber": display=f"₹{value:,.2f}"
        elif unit=="₹/unit": display=f"₹{value:,.2f}"
        else: display=f"{value:,.2f}"
        results.append({"Category":category,"Ratio":name,"Formula":formula,"Value":display,"Numeric value":value,
                        "Unit":unit,"Reporting period":_fy_label(year),"Source fields":fields,"Explanation":why,
                        "Status":status,"Missing inputs":missing,"Limitations":limitation})
    def v(k,y): return _at(fin,k,y)
    def ratio(cat,nm,formula,n,d,mult,unit,y,nfield,dfield,why,limitation="",allow_negative_d=False):
        value,reason=_safe(n,d,mult,allow_negative_d)
        miss=[]
        nlabel=FIELDS[nfield][2] if nfield in FIELDS else nfield
        dlabel=FIELDS[dfield][2] if dfield in FIELDS else dfield
        if n is None: miss.append(nlabel)
        if d is None: miss.append(dlabel)
        put(cat,nm,formula,value,unit,y,f"{nlabel}; {dlabel}",why,"; ".join(miss),limitation,reason)

    # Profitability / growth
    y=_latest_year(fin,["revenue","ebitda"]); rev=v("revenue",y); ebitda=v("ebitda",y); ebit=v("ebit",y)
    ratio("Profitability & Growth","EBITDA Margin","EBITDA / Revenue × 100",ebitda,rev,100,"%",y,"ebitda","revenue","EBITDA generated per rupee of revenue.")
    # EBIT margin duplicates the existing Operating Margin calculation when both fields are available.
    opmargin=(ebit/rev*100 if ebit is not None and rev not in (None,0) else None)
    put("Existing ratio audit","EBIT Margin (existing-equivalent)","EBIT / Revenue × 100",opmargin,"%",y,"Income: EBIT and Revenue","Equivalent to existing Operating Margin % when EBIT matches operating income.","","Alias only; not a duplicate ratio in the supplementary table.")
    yr=_latest_year(fin,["ebit","pretax","tax","debt","equity","cash"])
    roic=None; roic_missing=[]; roic_lim="NOPAT tax rate uses reported tax provision / pretax income; invested capital = debt + equity − cash."
    if yr:
        e,taxbase,tax,d,eq,c=[v(k,yr) for k in ["ebit","pretax","tax","debt","equity","cash"]]
        dp,ep,cp=[v(k,yr-1) for k in ["debt","equity","cash"]]
        allvals=[e,taxbase,tax,d,eq,c,dp,ep,cp]
        if any(x is None for x in allvals): roic_missing=[k for k,x in zip(["EBIT","Pretax Income","Tax Provision","Debt","Equity","Cash","Prior Debt","Prior Equity","Prior Cash"],allvals) if x is None]
        elif taxbase<=0: roic_missing=["positive Pretax Income required to derive effective tax rate"]
        else:
            nopat=e*(1-tax/taxbase); ic=d+eq-c; icp=dp+ep-cp; av=(ic+icp)/2
            roic,_=_safe(nopat,av,100)
            if roic is None: roic_missing=["average invested capital nonzero and positive"]
    put("Profitability & Growth","ROIC","NOPAT / Average Invested Capital × 100",roic,"%",yr,"Income: EBIT, pretax income, tax; Balance: debt, equity, cash (current/prior FY)","NOPAT uses effective tax rate; invested capital = debt + equity − cash.","; ".join(roic_missing),roic_lim)
    yi=_latest_year(fin,["ebit","pretax","tax","debt","equity","cash"]); incremental=None; imiss=[]; ilim="Change in invested capital must be nonzero; tax rates are period-specific."
    if yi:
        def nopat_ic(yy):
            e,tb,tax,d,eq,c=[v(k,yy) for k in ["ebit","pretax","tax","debt","equity","cash"]]
            if any(x is None for x in [e,tb,tax,d,eq,c]) or tb<=0:return None,None
            return e*(1-tax/tb), d+eq-c
        n0,ic0=nopat_ic(yi); n1,ic1=nopat_ic(yi-1)
        if None in [n0,ic0,n1,ic1]: imiss=["current/prior EBIT, pretax income, tax, debt, equity and cash"]
        else: incremental,why=_safe(n0-n1,ic0-ic1,100,allow_negative_d=True); imiss=[why] if incremental is None and why else []
    put("Profitability & Growth","Incremental Return on Capital","Change in NOPAT / Change in Invested Capital × 100",incremental,"%",yi,"Current/prior FY income and balance-sheet inputs","Year-on-year change in NOPAT divided by change in invested capital.","; ".join(imiss),ilim)
    def growth_ratio(label,metric):
        s=_metric_series(fin,metric); yy=int(max(s.index)) if not s.empty else None
        cur=_num(s.get(yy)) if yy else None; prev=_num(s.get(yy-1)) if yy else None
        value=None; reason=""
        if cur is not None and prev is not None:
            if prev==0: reason="previous-period value is zero"
            else: value=(cur-prev)/abs(prev)*100
        put("Profitability & Growth",label,"(Current − Previous) / |Previous| × 100",value,"%",yy,f"{FIELDS[metric][2]} — current and prior FY","Uses absolute previous-period denominator.","" if cur is not None and prev is not None else FIELDS[metric][2],"",reason)
    growth_ratio("Gross Profit Growth","gross_profit"); growth_ratio("EBITDA Growth","ebitda"); growth_ratio("EPS Growth","eps")

    # Liquidity / working capital
    yl=_latest_year(fin,["current_assets","current_liabilities"]); ca,cl=v("current_assets",yl),v("current_liabilities",yl); cash=v("cash",yl); short=v("short_investments",yl)
    cash_row=_find_row(fin.get("balance"),FIELDS["cash"][1]); cash_combined=bool(cash_row and "shortterminvestments" in _norm(cash_row))
    # The requested Cash Ratio excludes marketable securities. If the provider only exposes a
    # combined cash + short-term-investments line, do not mislabel that aggregate as cash-only.
    cash_only=None if cash_combined else cash
    ratio("Liquidity & Working Capital","Cash Ratio","(Cash + Cash Equivalents) / Current Liabilities",cash_only,cl,1,"x",yl,"cash","current_liabilities","Uses a cash-only balance-sheet field; a combined cash/investments field is insufficient.","Separate cash and equivalents unavailable when provider combines them with short-term investments." if cash_combined else "")
    wc=ca-cl if ca is not None and cl is not None else None
    put("Liquidity & Working Capital","Working Capital","Current Assets − Current Liabilities",wc,"₹ crore",yl,"Balance: Current Assets, Current Liabilities","Absolute working capital; values converted to ₹ crore for display.","" if wc is not None else "Current Assets; Current Liabilities")
    yw=_latest_year(fin,["current_assets","current_liabilities","revenue"]); ca0,cl0,rv0=[v(k,yw) for k in ["current_assets","current_liabilities","revenue"]]; ca1,cl1=[v(k,yw-1) for k in ["current_assets","current_liabilities"]] if yw else (None,None)
    avg_wc=((ca0-cl0)+(ca1-cl1))/2 if None not in [ca0,cl0,ca1,cl1] else None
    ratio("Liquidity & Working Capital","Working Capital to Sales","Average Working Capital / Revenue × 100",avg_wc,rv0,100,"%",yw,"current_assets/current_liabilities current and prior FY","revenue","Uses average working capital from beginning and ending balances.")
    yc=_latest_year(fin,["cfo","current_liabilities"]); ratio("Liquidity & Working Capital","Operating Cash Flow Ratio","CFO / Current Liabilities",v("cfo",yc),v("current_liabilities",yc),1,"x",yc,"cfo","current_liabilities","Operating cash flow coverage of current liabilities.")
    yd=_latest_year(fin,["cfo","debt"]); ratio("Liquidity & Working Capital","Cash Flow to Debt","CFO / Total Debt",v("cfo",yd),v("debt",yd),1,"x",yd,"cfo","debt","Operating cash flow relative to total debt.")
    ydint=_latest_year(fin,["cash","receivables","total_expenses","depreciation"]); csh,rec,expn,dep=[v(k,ydint) for k in ["cash","receivables","total_expenses","depreciation"]]; stinv=v("short_investments",ydint)
    # Approximation explicitly documented: total expenses minus D&A approximates cash operating expenses.
    cash_lbl=_find_row(fin.get("balance"),FIELDS["cash"][1]); cash_includes_st=bool(cash_lbl and "shortterminvestments" in _norm(cash_lbl))
    if cash_includes_st:
        defensive_assets=(csh+rec) if csh is not None and rec is not None else None
    else:
        defensive_assets=(csh+rec+stinv) if None not in [csh,rec,stinv] else None
    cash_exp=expn-abs(dep) if expn is not None and dep is not None else None
    defensive=None; defensive_reason=""
    if defensive_assets is not None and cash_exp is not None and cash_exp>0: defensive=defensive_assets/(cash_exp/365)
    elif cash_exp is not None: defensive_reason="cash operating expenses are zero or negative"
    put("Liquidity & Working Capital","Defensive Interval","(Cash + Marketable Securities + Receivables) / Average Daily Cash Operating Expenses",defensive,"days",ydint,"Balance: cash, short investments, receivables; Income: expenses; Cash Flow: D&A","365 days; uses total expenses less D&A as proxy when explicit cash operating expenses are unavailable.","" if defensive is not None else "; ".join(k for k,x in [("Cash",csh),("Receivables",rec),("Short-term Investments",stinv),("Expenses",expn),("D&A",dep)] if x is None),"Approximation: expenses less D&A may include non-cash items other than D&A.",defensive_reason)

    # Efficiency
    yi=_latest_year(fin,["cogs","inventory"]); cogs,inv=v("cogs",yi),v("inventory",yi); invp=v("inventory",yi-1) if yi else None; avginv=(inv+invp)/2 if inv is not None and invp is not None else None
    ratio("Operational Efficiency","Inventory Turnover","COGS / Average Inventory",cogs,avginv,1,"x",yi,"cogs","inventory current/prior FY","Uses average inventory, not ending inventory.")
    yr=_latest_year(fin,["credit_sales","receivables"]); cs=v("credit_sales",yr); rec=v("receivables",yr); recp=v("receivables",yr-1) if yr else None; avgrec=(rec+recp)/2 if rec is not None and recp is not None else None
    ratio("Operational Efficiency","Receivables Turnover","Credit Sales / Average Trade Receivables",cs,avgrec,1,"x",yr,"credit_sales","receivables current/prior FY","Does not substitute total revenue for credit sales.")
    yp=_latest_year(fin,["credit_purchases","payables"]); purch=v("credit_purchases",yp); pay=v("payables",yp); payp=v("payables",yp-1) if yp else None; avgpay=(pay+payp)/2 if pay is not None and payp is not None else None
    ratio("Operational Efficiency","Payables Turnover","Credit Purchases / Average Trade Payables",purch,avgpay,1,"x",yp,"credit_purchases","payables current/prior FY","Does not substitute COGS for credit purchases.")
    dso,_=_safe(avgrec,cs,365); dio,_=_safe(avginv,cogs,365); dpo,_=_safe(avgpay,purch,365)
    put("Operational Efficiency","Days Sales Outstanding (DSO)","Average Trade Receivables / Credit Sales × 365",dso,"days",yr,"Receivables average; Credit Sales","Requires disclosed credit sales; no revenue proxy.","Credit Sales or average receivables" if dso is None else "","")
    put("Operational Efficiency","Days Inventory Outstanding (DIO)","Average Inventory / COGS × 365",dio,"days",yi,"Inventory current/prior; COGS","Uses average inventory.","COGS or average inventory" if dio is None else "","")
    put("Operational Efficiency","Days Payables Outstanding (DPO)","Average Trade Payables / Credit Purchases × 365",dpo,"days",yp,"Payables current/prior; Credit Purchases","Requires disclosed credit purchases.","Credit Purchases or average payables" if dpo is None else "","")
    ccc=dso+dio-dpo if None not in [dso,dio,dpo] else None
    put("Operational Efficiency","Cash Conversion Cycle","DSO + DIO − DPO",ccc,"days",max([x for x in [yr,yi,yp] if x is not None],default=None),"Calculated DSO, DIO, DPO","Only shown when all three components are available.","; ".join(x for x,z in [("DSO",dso),("DIO",dio),("DPO",dpo)] if z is None))
    yfasset=_latest_year(fin,["revenue","ppe"]); ppe,ppep=v("ppe",yfasset),v("ppe",yfasset-1) if yfasset else (None,None)
    # tuple assignment guard
    if yfasset: ppe,ppep=v("ppe",yfasset),v("ppe",yfasset-1)
    avgppe=(ppe+ppep)/2 if ppe is not None and ppep is not None else None
    ratio("Operational Efficiency","Fixed Asset Turnover","Revenue / Average Net Fixed Assets",v("revenue",yfasset),avgppe,1,"x",yfasset,"revenue","ppe current/prior FY","Uses average net PPE.")

    # Cash-flow quality
    ycf=_latest_year(fin,["cfo","capex","net_income","revenue"]); cfo,ni,rev=[v(k,ycf) for k in ["cfo","net_income","revenue"]]
    put("Existing ratio audit","CFO to Net Income (existing-equivalent)","CFO / Net Income",_safe(cfo,ni)[0],"x",ycf,"Existing Cash Conversion; CFO and Net Income","Alias for existing Cash Conversion ratio; not duplicated in main matrix.","","Alias only; not a new matrix row.")
    capex=v("capex",ycf); capex_spend=abs(capex) if capex is not None else None; fcf=cfo-capex_spend if cfo is not None and capex_spend is not None else None
    ratio("Cash-Flow Quality","Free Cash Flow Margin","FCF / Revenue × 100",fcf,rev,100,"%",ycf,"CFO and Capex (absolute spend)","revenue","FCF = CFO − absolute capex spend, robust to provider sign conventions.")
    mcap=_num(info.get("marketCap")); ratio("Cash-Flow Quality","Free Cash Flow Yield","FCF / Market Capitalization × 100",fcf,mcap,100,"%",ycf,"CFO and Capex","marketCap","Uses latest market cap against latest reported full-year FCF.","Market cap date differs from FY period by design.")
    ycs=_latest_year(fin,["capex","revenue"]); ratio("Cash-Flow Quality","Capex to Sales","Absolute Capex / Revenue × 100",abs(v("capex",ycs)) if v("capex",ycs) is not None else None,v("revenue",ycs),100,"%",ycs,"capex","revenue","Capex expressed as spending magnitude.")
    ycc=_latest_year(fin,["capex","cfo"]); ratio("Cash-Flow Quality","Capex to CFO","Absolute Capex / CFO × 100",abs(v("capex",ycc)) if v("capex",ycc) is not None else None,v("cfo",ycc),100,"%",ycc,"capex","cfo","Negative or zero CFO makes interpretation limited.",allow_negative_d=True)
    yce=_latest_year(fin,["cfo","ebitda"]); ratio("Cash-Flow Quality","Cash Flow to EBITDA","CFO / EBITDA",v("cfo",yce),v("ebitda",yce),1,"x",yce,"cfo","ebitda","Measures cash conversion from EBITDA.")
    ya=_latest_year(fin,["net_income","cfo","assets"])
    if ya is not None:
        ai,ap=v("assets",ya),v("assets",ya-1)
    else:
        ai,ap=None,None
    avga=(ai+ap)/2 if ai is not None and ap is not None else None; ni,cfo=v("net_income",ya),v("cfo",ya)
    ratio("Cash-Flow Quality","Accrual Ratio","(Net Income − CFO) / Average Total Assets × 100",(ni-cfo) if ni is not None and cfo is not None else None,avga,100,"%",ya,"net_income and cfo","assets current/prior FY","Uses average total assets.")

    # Valuation / per-share
    yv=max(_metric_series(fin,"revenue").index) if not _metric_series(fin,"revenue").empty else None
    rev=v("revenue",yv); shares=_num(info.get("sharesOutstanding")); shares_src="Provider current shares outstanding"
    stmt_shares=v("shares",yv)
    if stmt_shares is not None: shares=stmt_shares; shares_src="Balance: shares outstanding in matching FY"
    ratio("Valuation & Per-Share","Price-to-Sales (P/S)","Market Capitalization / Annual Revenue",mcap,rev,1,"x",yv,"marketCap","revenue","Market cap uses latest provider value and annual revenue uses latest full FY.","Market cap date may differ from FY period.")
    ratio("Valuation & Per-Share","Earnings Yield","Annual EPS / Share Price × 100",v("eps",yv),px,100,"%",yv,"Reported EPS","Latest share price","Annual reported EPS divided by latest share price.","Price and annual EPS are from different as-of dates.")
    debt=v("debt",yv); cash=v("cash",yv); pref=v("preferred",yv); nci=v("nci",yv)
    ev=None; evmissing=[name for name,x in [("Market Capitalization",mcap),("Total Debt",debt),("Preferred Equity",pref),("Non-controlling Interests",nci),("Cash and Cash Equivalents",cash)] if x is None]
    if not evmissing: ev=mcap+debt+pref+nci-cash
    ratio("Valuation & Per-Share","Enterprise Value / Sales","Enterprise Value / Annual Revenue",ev,rev,1,"x",yv,"Market cap, debt, preferred equity, non-controlling interest, cash","revenue","Strict EV formula; missing components are not assumed zero.","Missing EV components: "+", ".join(evmissing) if evmissing else "")
    put("Valuation & Per-Share","Enterprise Value","Market Cap + Debt + Preferred Equity + Non-controlling Interests − Cash",ev,"₹ crore",yv,"Market data: market cap; Balance: debt, preferred equity, NCI, cash","Strict formula only; does not silently substitute missing components.","; ".join(evmissing),"Market cap converted to INR units; formula uses same raw INR-scale provider amounts.")
    equity=v("equity",yv)
    ratio("Valuation & Per-Share","Book Value per Share","Equity Attributable to Ordinary Shareholders / Shares Outstanding",equity,shares,1,"₹/share",yv,"equity","shares","Uses common/shareholders equity and share count.","Current provider shares used only if period-matched shares unavailable." if stmt_shares is None else "")
    goodwill=v("goodwill",yv); intangible=v("intangibles",yv); tbv=(equity-goodwill-intangible) if None not in [equity,goodwill,intangible] else None
    tbvps,_=_safe(tbv,shares)
    put("Valuation & Per-Share","Tangible Book Value per Share","(Equity − Goodwill − Other Intangibles) / Shares Outstanding",tbvps,"₹/share",yv,"Balance: common equity, goodwill, other intangibles, shares","No missing goodwill/intangibles are assumed zero.","; ".join(n for n,x in [("Equity",equity),("Goodwill",goodwill),("Other Intangible Assets",intangible),("Shares Outstanding",shares)] if x is None))
    ptbv,_=_safe(px,tbvps)
    put("Valuation & Per-Share","Price-to-Tangible Book","Share Price / Tangible Book Value per Share",ptbv,"x",yv,"Market: latest price; Balance: equity, goodwill, intangibles and shares","Only meaningful for positive tangible book value per share.","Share price or tangible book/share" if ptbv is None else "","N/A if tangible book value/share is non-positive.")
    peg,peg_reason=calculate_peg(fin,info)
    epss=_metric_series(fin,"eps"); epsy=int(max(epss.index)) if not epss.empty else None
    put("Valuation & Per-Share","PEG (calculated; replaces vendor PEG)","Trailing P/E / EPS growth percentage points",peg,"x",epsy,"Market: trailing P/E; Income: current/prior annual reported EPS","Uses percentage-point convention: 20% growth is entered as 20.","; ".join([peg_reason] if peg_reason else []),"Replaces opaque vendor PEG in main matrix when applicable.")

    # Dividends / shareholder returns
    dps_by_fy={}
    if isinstance(dividends,pd.Series) and len(dividends):
        for date, amount in dividends.items():
            val0=_num(amount)
            if val0 is not None:
                y0=_fy_end_year(date); dps_by_fy[y0]=dps_by_fy.get(y0,0.0)+val0
    dy=max(dps_by_fy) if dps_by_fy else None; dps=dps_by_fy.get(dy) if dy else None; dps_prev=dps_by_fy.get(dy-1) if dy else None
    put("Dividend & Shareholder","Dividend per Share (DPS)","Sum of actual dividend-per-share events during Indian FY",dps,"₹/share",dy,"Provider dividend events and ex-dates","Uses actual corporate-action dividend events; FY uses Apr–Mar.","Dividend event history" if dps is None else "")
    dg=(dps-dps_prev)/abs(dps_prev)*100 if dps is not None and dps_prev not in [None,0] else None
    put("Dividend & Shareholder","Dividend Growth","(Current DPS − Previous DPS) / |Previous DPS| × 100",dg,"%",dy,"Dividend events: current/prior FY","Special, interim and final dividends are summed together.","Previous FY DPS" if dps_prev is None else "Previous DPS is zero" if dps_prev==0 else "")
    yd=_latest_year(fin,["net_income","dividends_paid"]); ni,div_paid=v("net_income",yd),v("dividends_paid",yd); retention=(ni-abs(div_paid))/ni*100 if ni is not None and div_paid is not None and ni>0 else None
    put("Dividend & Shareholder","Retention Ratio","(Net Income − Ordinary Dividends) / Net Income × 100",retention,"%",yd,"Income: net income; Cash flow: dividends paid","Dividends paid normalized to cash-outflow magnitude.","; ".join([n for n,x in [("Net Income",ni),("Dividends Paid",div_paid)] if x is None]),"Cash dividends paid may differ from declared dividends or ordinary-only dividends.")
    buy=v("buybacks",yd); issuance=v("share_issuance",yd)
    sh_yield=None
    if None not in [div_paid,buy,issuance,mcap] and mcap!=0: sh_yield=(abs(div_paid)+abs(buy)-abs(issuance))/mcap*100
    put("Dividend & Shareholder","Shareholder Yield","(Dividends + Net Buybacks − Net Share Issuance) / Market Cap × 100",sh_yield,"%",yd,"Cash flow: dividends, share buybacks, share issuance; Market: market cap","Normalizes provider cash-outflow signs using absolute amounts.","; ".join([n for n,x in [("Dividends Paid",div_paid),("Share Buybacks",buy),("Share Issuance",issuance),("Market Capitalization",mcap)] if x is None]),"Current market cap used against annual flows.")

    # Solvency
    yl=_latest_year(fin,["debt","cash","equity"]); debt,cash,eq=[v(k,yl) for k in ["debt","cash","equity"]]
    ratio("Financial Strength & Risk","Net Debt to Equity","(Total Debt − Cash) / Equity",(debt-cash) if debt is not None and cash is not None else None,eq,1,"x",yl,"debt and cash","equity","Negative net debt indicates net cash.")
    ratio("Financial Strength & Risk","Total Liabilities to Equity","Total Liabilities / Equity",v("liabilities",yl),eq,1,"x",yl,"liabilities","equity","Uses total liabilities from the same FY.")
    ylt=_latest_year(fin,["long_debt","equity"]); ratio("Financial Strength & Risk","Long-term Debt to Equity","Long-term Debt / Equity",v("long_debt",ylt),v("equity",ylt),1,"x",ylt,"long_debt","equity","Requires a separately disclosed long-term debt field.")
    yds=_latest_year(fin,["cash_available_debt_service","principal_repayments","interest_paid"]); cad,pr,ip=[v(k,yds) for k in ["cash_available_debt_service","principal_repayments","interest_paid"]]; dscr,_=_safe(cad,abs(pr)+abs(ip) if pr is not None and ip is not None else None)
    put("Financial Strength & Risk","Debt Service Coverage Ratio (DSCR)","Cash Available for Debt Service / (Principal Repayments + Interest Due)",dscr,"x",yds,"Cash available for debt service, principal repayments, interest due","Does not substitute CFO for cash available for debt service.","; ".join([n for n,x in [("Cash Available for Debt Service",cad),("Principal Repayments",pr),("Interest Paid",ip)] if x is None]))

    # Sector-specific ratios. Non-eligible businesses are marked not applicable, not N/A.
    industry_only=str(info.get("industry") or "").lower()
    isbank=any(x in industry_only for x in ["bank","commercial banks","regional banks"]) or "bank" in str(sector or "").lower()
    isnf=any(x in sector_text for x in ["bank","nbfc","financial services","financial service","consumer finance"])
    isins=any(x in sector_text for x in ["insurance"])
    ispharma=any(x in sector_text for x in ["pharma","pharmaceutical","drug manufacturers","biotechnology"])
    ishospital=any(x in sector_text for x in ["hospital","healthcare providers"])
    isit=any(x in sector_text for x in ["information technology","software","it services"])
    istele=any(x in sector_text for x in ["telecom","telecommunication"])
    ismetals=any(x in sector_text for x in ["metal","mining","commodit"])
    def sect_ratio(eligible, cat, nm, formula, nfield, dfield, multiplier, unit, explain):
        if not eligible:
            put(cat,nm,formula,None,unit,None,"Company sector / industry classification",explain,"","Not applicable — company outside this sector scope")
        else:
            yy=_latest_year(fin,[nfield,dfield]); ratio(cat,nm,formula,v(nfield,yy),v(dfield,yy),multiplier,unit,yy,nfield,dfield,explain)
    sect_ratio(isbank,"Sector-Specific — Banks","Net Interest Margin","Net Interest Income / Average Earning Assets × 100","net_interest_income","earning_assets",100,"%","Requires reported NII and average earning assets.")
    sect_ratio(isbank,"Sector-Specific — Banks","CASA Ratio","CASA Deposits / Total Deposits × 100","casa","deposits",100,"%","Requires CASA and total deposit disclosures.")
    sect_ratio(isbank,"Sector-Specific — Banks","Gross NPA","Gross NPA / Gross Advances × 100","gross_npa","gross_advances",100,"%","Requires bank regulatory disclosures.")
    sect_ratio(isbank,"Sector-Specific — Banks","Net NPA","Net NPA / Net Advances × 100","net_npa","net_advances",100,"%","Requires bank regulatory disclosures.")
    sect_ratio(isbank,"Sector-Specific — Banks","Provision Coverage Ratio","NPA Provisions / Gross NPA × 100","npa_provisions","gross_npa",100,"%","Requires reported NPA provisions and gross NPAs.")
    sect_ratio(isnf,"Sector-Specific — Banks/NBFCs","Capital Adequacy Ratio","Eligible Regulatory Capital / Risk-Weighted Assets × 100","reg_capital","rwa",100,"%","Requires regulatory capital and risk-weighted assets.")
    sect_ratio(isins,"Sector-Specific — Insurance","Solvency Ratio","Available Solvency Margin / Required Solvency Margin","solvency_available","solvency_required",1,"x","Requires statutory solvency inputs.")
    if isins:
        yc=_latest_year(fin,["claims","insurance_expenses","earned_premiums"]); claims,iexp,prem=[v(k,yc) for k in ["claims","insurance_expenses","earned_premiums"]]; comb,_=_safe(claims+iexp if claims is not None and iexp is not None else None,prem,100)
        put("Sector-Specific — Insurance","Combined Ratio","(Claims Incurred + Relevant Expenses) / Earned Premiums × 100",comb,"%",yc,"Claims, relevant insurance expenses, earned premiums","Combined ratio above 100% implies underwriting loss before investment income.","; ".join(n for n,x in [("Claims Incurred",claims),("Insurance Expenses",iexp),("Earned Premiums",prem)] if x is None))
    else: put("Sector-Specific — Insurance","Combined Ratio","(Claims Incurred + Relevant Expenses) / Earned Premiums × 100",None,"%",None,"Company classification","Not applicable","","Not applicable — company outside this sector scope")
    sect_ratio(ispharma,"Sector-Specific — Healthcare/Pharma","R&D Intensity","R&D Expense / Revenue × 100","rd","revenue",100,"%","Requires R&D expense and revenue disclosures.")
    sect_ratio(ishospital,"Sector-Specific — Hospitals","Revenue per Bed","Revenue / Average Operational Beds","revenue","beds",1,"₹/bed","Requires average operational beds; standard feeds often lack this.")
    # Revenue/employee requires average employee count, not a single period count.
    ye=_latest_year(fin,["employees","revenue"])
    if ye is not None:
        e0,e1=v("employees",ye),v("employees",ye-1)
    else:
        e0,e1=None,None
    ave=(e0+e1)/2 if e0 is not None and e1 is not None else None
    if isit:
        ratio("Sector-Specific — IT Services","Revenue per Employee","Revenue / Average Employee Count",v("revenue",ye),ave,1,"₹/employee",ye,"revenue","employees current/prior FY","Uses two-year average headcount.")
        sect_ratio(True,"Sector-Specific — IT Services","Employee Utilization","Billable Employee Hours / Available Employee Hours × 100","billable_hours","available_hours",100,"%","Requires disclosed billable and available hours.")
    else:
        put("Sector-Specific — IT Services","Revenue per Employee","Revenue / Average Employee Count",None,"₹/employee",None,"Company sector","Not applicable","","Not applicable — company outside this sector scope")
        put("Sector-Specific — IT Services","Employee Utilization","Billable Employee Hours / Available Employee Hours × 100",None,"%",None,"Company sector","Not applicable","","Not applicable — company outside this sector scope")
    sect_ratio(istele,"Sector-Specific — Telecom","ARPU","Telecom Service Revenue / Average Subscribers","telecom_revenue","subscribers",1,"₹/subscriber","Requires comparable service revenue and average subscriber count.")
    sect_ratio(ismetals,"Sector-Specific — Metals/Commodities","EBITDA per Unit","EBITDA / Units Sold","ebitda","units_sold",1,"₹/unit","Requires EBITDA and units sold measured on a compatible basis.")

    altz,alt_formula,alt_year,alt_missing,alt_lim=calculate_altman_z(fin,info,sector)
    put("Model Audit","Altman Z-Score (validated variant)",alt_formula,altz,"score",alt_year,"Balance: working capital, assets, retained earnings, total liabilities; Income: EBIT/revenue; Market: market equity at matching FY-end","Model variant selected only when company type and required inputs are identified.","; ".join(alt_missing),alt_lim)
    fscore,fnotes=calculate_piotroski(fin)
    fmissing=[] if np.isfinite(fscore) else fnotes
    fy_pi=_latest_year(fin,["net_income","cfo","assets","long_debt","current_assets","current_liabilities","revenue","gross_profit","shares"])
    put("Model Audit","Piotroski F-Score (nine tests)","Sum of nine financial-strength binary tests (0–9)",fscore,"score / 9",fy_pi,"Income: net income, revenue, gross profit; Cash Flow: CFO; Balance: assets, long-term debt, current assets/liabilities, shares","Reports only when all nine current/prior-year tests can be calculated.","; ".join(fmissing),"Share-count change is a proxy for dilution; validate source share-count consistency.")

    # Add a separate status inventory for existing main matrix ratios (no duplicates in this panel).
    df=pd.DataFrame(results)
    if not df.empty:
        df=df.drop_duplicates(subset=["Category","Ratio"],keep="first")
    return df


EXISTING_FORMULAS = {
    "Revenue Growth %": ("(Current Revenue − Previous Revenue) / |Previous Revenue| × 100", "Income: Revenue current/prior FY", "Growth metric; previous revenue should be nonzero."),
    "Net Profit Growth %": ("(Current Net Income − Previous Net Income) / |Previous Net Income| × 100", "Income: Net Income current/prior FY", "Growth metric; previous net income should be nonzero."),
    "Gross Margin %": ("Gross Profit / Revenue × 100", "Income: Gross Profit, Revenue", "Gross profit margin."),
    "Operating Margin %": ("Operating Income / Revenue × 100", "Income: Operating Income, Revenue", "Operating margin; may be EBIT equivalent."),
    "Net Margin %": ("Net Income / Revenue × 100", "Income: Net Income, Revenue", "Net profit margin."),
    "ROE %": ("Net Income / Average Equity × 100", "Income: Net Income; Balance: current/prior equity", "Average equity should be used."),
    "ROA %": ("Net Income / Average Assets × 100", "Income: Net Income; Balance: current/prior total assets", "Average total assets should be used."),
    "ROCE %": ("EBIT / (Average Total Assets − Current Liabilities) × 100", "Income: EBIT; Balance: assets/liabilities", "Existing app formula; uses latest current liabilities."),
    "Current Ratio": ("Current Assets / Current Liabilities", "Balance: Current Assets, Current Liabilities", "Liquidity measure."),
    "Quick Ratio": ("(Current Assets − Inventory − Prepaids) / Current Liabilities", "Balance: current assets, inventory/prepaids, current liabilities", "Available data may not fully identify all non-quick assets."),
    "Debt / Equity": ("Total Debt / Shareholders' Equity", "Balance: Total Debt, Equity", "Uses ending balances."),
    "Debt / Assets %": ("Total Debt / Total Assets × 100", "Balance: Total Debt, Total Assets", "Leverage measure."),
    "Net Debt (₹ cr)": ("Total Debt − Cash", "Balance: Total Debt, Cash", "Amount, not a ratio; displayed in ₹ crore."),
    "Interest Coverage": ("EBIT / |Interest Expense|", "Income: EBIT, Interest Expense", "Interest expense magnitude used."),
    "Debt / EBITDA": ("Total Debt / EBITDA", "Balance: Total Debt; Income: EBITDA", "Leverage multiple."),
    "Asset Turnover": ("Revenue / Average Total Assets", "Income: Revenue; Balance: current/prior assets", "Average assets should be used."),
    "Equity Multiplier": ("Average Total Assets / Average Equity", "Balance: current/prior assets/equity", "Leverage multiplier."),
    "CFO (₹ cr)": ("Reported Cash Flow from Operations", "Cash Flow: CFO", "Amount, not a ratio."),
    "Capex (₹ cr)": ("Reported Capital Expenditure", "Cash Flow: Capex", "Amount, not a ratio; sign is provider-dependent."),
    "FCF (₹ cr)": ("CFO + Reported Capex", "Cash Flow: CFO, Capex", "Existing app calculation follows signed cash-flow convention."),
    "Cash Conversion": ("CFO / Net Income", "Cash Flow: CFO; Income: Net Income", "Equivalent to CFO-to-net-income ratio."),
    "PE": ("Provider-reported trailing P/E", "Market-data provider info", "Vendor field; definition/as-of can vary."),
    "Forward PE": ("Provider-reported forward P/E", "Market-data provider info", "Estimate-based vendor field."),
    "PB": ("Provider-reported price-to-book", "Market-data provider info", "Vendor field; should be checked against statement equity."),
    "EV/EBITDA": ("Provider-reported EV / EBITDA", "Market-data provider info", "Vendor field; component treatment can vary."),
    "PEG": ("Trailing P/E / Annual EPS Growth (%) in percentage points", "Market: trailing P/E; Income: annual reported EPS current/prior FY", "This field is recalculated by InvestorLens; zero/negative EPS growth yields N/A."),
    "Dividend Yield %": ("Provider-reported dividend yield × 100", "Market-data provider info", "Can use indicated/trailing dividend policy; validate against dividend history."),
    "Payout Ratio %": ("Provider-reported payout ratio × 100", "Market-data provider info", "Vendor definition may differ by provider."),
    "Market Cap (₹ cr)": ("Provider market capitalization", "Market-data provider info", "Displayed in ₹ crore."),
    "Beta (vendor)": ("Provider-reported beta", "Market-data provider info", "Window/index methodology may vary by provider."),
}


EXISTING_UNITS = {
    "Revenue Growth %":"%", "Net Profit Growth %":"%", "Gross Margin %":"%", "Operating Margin %":"%",
    "Net Margin %":"%", "ROE %":"%", "ROA %":"%", "ROCE %":"%", "Current Ratio":"x", "Quick Ratio":"x",
    "Debt / Equity":"x", "Debt / Assets %":"%", "Net Debt (₹ cr)":"₹ crore", "Interest Coverage":"x", "Debt / EBITDA":"x",
    "Asset Turnover":"x", "Equity Multiplier":"x", "CFO (₹ cr)":"₹ crore", "Capex (₹ cr)":"₹ crore", "FCF (₹ cr)":"₹ crore",
    "Cash Conversion":"x", "PE":"x", "Forward PE":"x", "PB":"x", "EV/EBITDA":"x", "PEG":"x",
    "Dividend Yield %":"%", "Payout Ratio %":"%", "Market Cap (₹ cr)":"₹ crore", "Beta (vendor)":"x"
}


def existing_ratio_audit(current_ratios: Optional[Mapping[str,Any]], fin: Mapping[str,Any]) -> pd.DataFrame:
    values = dict(current_ratios or {})
    periods = []
    for field in ["revenue", "net_income", "assets", "equity"]:
        series = _metric_series(fin, field)
        if not series.empty: periods.append(int(max(series.index)))
    latest = max(periods) if periods else None
    rows=[]
    for name in sorted(EXISTING):
        formula,fields,notes = EXISTING_FORMULAS.get(name,("Existing calculation/vendor field","Existing calc_ratios() or provider info","Formula not explicitly exposed by original UI; see app.py calc_ratios()."))
        value=values.get(name,np.nan)
        if name=="PEG": value=values.get("PEG",np.nan)
        try:
            value=float(value)
            if not np.isfinite(value): value=NA
        except Exception: value=NA
        rows.append({"Existing ratio":name,"Value":value,"Unit":EXISTING_UNITS.get(name,""),"Formula / provider field":formula,"Reporting period":_fy_label(latest),"Source fields":fields,"Audit note":notes,"Status":"Available in original matrix" if value!=NA else NA})
    return pd.DataFrame(rows)


def render_ratio_audit_panel(fin, info, price, dividends, sector, current_ratios=None):
    import streamlit as st
    st.subheader("Supplemental ratios & calculation audit")
    st.caption("Existing ratio-matrix entries are not duplicated. Each row shows formula, value or N/A, FY, source fields, missing inputs and limitations. Financial calculations use raw matching-period statement amounts; monetary display is ₹ crore.")
    data=calculate_ratio_audit(fin,info,price,dividends,sector,current_ratios)
    cats=list(data["Category"].drop_duplicates()) if not data.empty else []
    for cat in cats:
        if cat == "Existing ratio audit":
            continue
        visible=data[data["Category"]==cat]
        if cat == "Valuation & Per-Share":
            visible=visible[visible["Ratio"]!="PEG (calculated; replaces vendor PEG)"]
        with st.expander(cat, expanded=cat in ["Profitability & Growth","Liquidity & Working Capital","Cash-Flow Quality"]):
            st.dataframe(visible.drop(columns=["Numeric value"],errors="ignore"),use_container_width=True,hide_index=True)
    with st.expander("Audit of ratios already in the original matrix (no duplicate calculations)"):
        st.dataframe(existing_ratio_audit(current_ratios,fin),use_container_width=True,hide_index=True)
    st.download_button("Download supplemental ratio audit CSV",data.to_csv(index=False).encode("utf-8"),file_name="investorlens_supplemental_ratios.csv",mime="text/csv")
