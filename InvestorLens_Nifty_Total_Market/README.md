# InvestorLens — NIFTY Total Market + Audited Ratios + Live Industry Dashboard

This version is based on the supplied `InvestorLens_Nifty_Total_Market(2).zip` source project. Existing financial, CAPM, recommendation, technical, sentiment, 5-year statements, dividend, forecast, screener and data-audit tabs are retained. Two tabs have been appended:

- **Supplemental Ratios Audit** — ratios missing from the original matrix, formula/source-field/period/unit reporting, missing-data reasons and a separate audit of the original 30 ratios (not duplicated in the main matrix).
- **Live Market & Industry** — automatic provider-backed market movers, green/red NIFTY sector-index charts, 1D/5D/1M/3M/6M/1Y/3Y/5Y return periods, NIFTY 500-relative performance, industry peer averages and a Positive/Neutral/Caution analytical signal.

## Run on Windows

1. Extract the ZIP.
2. Open PowerShell in the extracted `InvestorLens_v4_5FY_Financials_fixed` folder. (In File Explorer, click the address bar, type `powershell`, press Enter.)
3. Create/activate an environment and install requirements:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Open `http://localhost:8501` if the browser does not open automatically.

The `start.bat` and `start.sh` launchers remain available. If Windows Smart App Control blocks `start.bat`, use the PowerShell commands above instead of disabling Windows security.

## Optional market polling configuration

Set these environment variables before launching the app if required by your provider/data entitlement:

```text
MARKET_POLL_SECONDS=300
MARKET_DATA_WORKERS=8
FINNHUB_API_KEY=
```

The industry/market panel automatically updates on the interval, without a manual refresh button or a busy `while` loop. The current adapter does not expose a documented WebSocket subscription, so it uses polling of the existing quote/history functions. The default interval is 300 seconds and can be configured to comply with the selected provider's terms/rate limits.

## Data source and live-data limitation

- The stock universe loader attempts to retrieve the official NIFTY Total Market constituent CSV and retains the bundled universe as an offline fallback.
- Stock prices, percent changes, volume and stock history use the existing InvestorLens adapters (optional configured Finnhub quote call, then Yahoo Finance/yfinance fallback).
- The industry charts use available historical prices for named NIFTY sector indices; unsupported or unavailable index histories are omitted, not replaced with synthetic returns.
- “Most active” uses the provider quote volume if exposed; otherwise it labels the latest daily-bar volume. The current adapter cannot guarantee intraday cumulative volume or exchange-grade realtime data.
- Yahoo Finance/yfinance does not guarantee exchange-grade real-time NSE prices or complete historical fundamentals. For a public/commercial investment product, integrate an authorized/licensed data vendor or broker feed and honor its polling/WebSocket terms.

Official list of NIFTY sectoral indices: https://www.niftyindices.com/indices/equity/sectoral-indices

## Financial ratio audit

The original matrix remains in place. Supplementary rows are added only for formulas not already calculated in the original matrix. Existing metrics including Revenue Growth, Net Profit Growth, Gross Margin, Operating Margin, Net Margin, ROE, ROA, ROCE, Current/Quick Ratio, Debt/Equity, Debt/Assets, Net Debt amount, Interest Coverage, Debt/EBITDA, Asset Turnover, Equity Multiplier, CFO/Net Income (Cash Conversion), CFO/Capex/FCF amounts, P/E, Forward P/E, P/B, EV/EBITDA, PEG, Dividend Yield, Payout Ratio, Market Cap and vendor Beta are documented in the existing-ratio audit expander instead of duplicated.

PEG is recalculated into the original PEG field using trailing P/E divided by the reported year-on-year annual EPS growth rate in percentage points; zero, negative or missing EPS growth yields N/A.

All supplemental results include formula, value or `N/A — insufficient data`, units, financial year, source fields, explanation, missing inputs and limitation. Monetary statement arithmetic stays in the provider's raw units; amount results are formatted in INR crore only for display. Per-share metrics remain INR/share. The current-period market cap is explicitly called out where it is combined with annual financials.

The engine intentionally marks the following as unavailable when the required fields are absent: credit sales, credit purchases, separate cash-only balances when the source combines cash and short-term investments, period-matched market equity for Altman Z, preferred equity/non-controlling interest for strict Enterprise Value, operating beds, average headcount/hours, subscriber counts, units sold, bank regulatory ratios, insurance solvency/claims disclosures and debt-service cash available. It does not infer these values from unrelated fields.

Piotroski F-Score is calculated only when all nine binary tests have consecutive annual inputs; otherwise it reports N/A and lists missing fields. Altman Z is calculated only when company model eligibility and period-matched market equity at FY-end are available. The prior incomplete Altman proxy has been retired.

## Test suite

Run:

```powershell
python -m unittest discover -s tests -v
```

The included 30 unit tests validate calculation arithmetic, period logic, denominator safeguards, missing-input behavior, strict enterprise-value requirements, PEG's percentage-point convention, complete nine-test Piotroski gating, sector applicability and genuine-history-based industry return math. Unit-test fixtures are synthetic and are used only for calculation tests; they are never used as app market prices or financial values.

See `COMPLETION_REPORT.md` for the coverage table and verification limitations.
