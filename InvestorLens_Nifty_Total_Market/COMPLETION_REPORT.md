# InvestorLens upgrade — completion and validation report

## Source reviewed

The supplied `InvestorLens_Nifty_Total_Market(2).zip` was extracted and the existing `app.py`, statement adapter, ratio calculator, tab list, quote adapter, and data-universe loader were inspected. The changes are merged into this full project so existing screens are retained.

## Changes integrated

1. Added a **Supplemental Ratios Audit** tab after the original tabs. It displays formula, calculated value or `N/A — insufficient data`, unit, Indian FY period, input/source fields, explanation, missing inputs and limitation.
2. Existing ratio entries are not inserted again into the main ratio matrix. A separate audit expander documents the 30 existing matrix metrics and their formulas/source fields.
3. Recalculated the existing PEG field using trailing P/E / reported year-on-year EPS growth in percentage points (`20%` growth means divide P/E by `20`, not `0.20`). Zero, negative or missing EPS growth yields N/A.
4. Added a **Live Market & Industry** tab using existing quote/history/financial callbacks. It auto-polls with `st.fragment(run_every=...)`, without a `while` loop.
5. Added top gainers, top losers and most-active tables, industry performance green/red bar charts, 1D/5D/1M/3M/6M/1Y/3Y/5Y filters, comparison to NIFTY 500 where supported, company-to-peer average comparisons and a transparent Positive/Neutral/Caution signal.
6. Added stock quote volume support from the provider's quote object where returned. If unavailable, the market panel uses the latest daily-bar volume and labels the source; it does not present it as verified intraday cumulative volume.
7. Retired the old incomplete Altman proxy. Altman Z now requires a suitable company-type model and period-matched FY-end market equity; otherwise it is N/A. Piotroski now requires all nine binary tests and current/prior-year input fields; otherwise it is N/A with missing-field notes.
8. Kept provider data external; no company price, statement, sector return or ratio is filled with fake values.

## Ratio inventory

### Already present in the original matrix (30 entries; retained, not duplicated)

Revenue Growth %, Net Profit Growth %, Gross Margin %, Operating Margin %, Net Margin %, ROE %, ROA %, ROCE %, Current Ratio, Quick Ratio, Debt / Equity, Debt / Assets %, Net Debt (₹ cr), Interest Coverage, Debt / EBITDA, Asset Turnover, Equity Multiplier, CFO (₹ cr), Capex (₹ cr), FCF (₹ cr), Cash Conversion, PE, Forward PE, PB, EV/EBITDA, PEG, Dividend Yield %, Payout Ratio %, Market Cap (₹ cr), Beta (vendor).

`EBIT Margin` is documented as an existing-equivalent of Operating Margin only when the provider EBIT field matches Operating Income; it is not duplicated as a new ratio. CFO/Net Income is documented as an existing-equivalent of Cash Conversion, not duplicated. PEG's value in the original field is recalculated transparently.

### Newly added requested calculations

EBITDA Margin; ROIC; Incremental Return on Capital; Gross Profit Growth; EBITDA Growth; EPS Growth; Cash Ratio; Working Capital; Working Capital to Sales; Operating Cash Flow Ratio; Cash Flow to Debt; Defensive Interval; Inventory Turnover; Receivables Turnover; Payables Turnover; DSO; DIO; DPO; Cash Conversion Cycle; Fixed Asset Turnover; FCF Margin; FCF Yield; Capex to Sales; Capex to CFO; Cash Flow to EBITDA; Accrual Ratio; P/S; Earnings Yield; EV/Sales; strict Enterprise Value; Book Value per Share; Tangible Book Value per Share; Price-to-Tangible Book; Dividend per Share; Dividend Growth; Shareholder Yield; Retention Ratio; Net Debt to Equity; Total Liabilities to Equity; Long-term Debt to Equity; DSCR.

Conditional models/sector metrics also added: Altman Z variant gating, complete-input Piotroski F-Score, bank NIM/CASA/Gross NPA/Net NPA/Provision Coverage, bank/NBFC capital adequacy, insurance solvency/combined ratio, healthcare/pharma R&D intensity, hospital revenue per bed, IT revenue per employee/utilization, telecom ARPU, metals EBITDA per unit.

### Explicitly unavailable when missing inputs

- Credit Sales and Credit Purchases: summary statement APIs usually do not expose them; total revenue/COGS is not substituted.
- Cash Ratio: N/A when only a combined cash + short-term-investments value is supplied and a cash-only value cannot be separated.
- Defensive Interval: requires cash, receivables, marketable securities and expense/depreciation inputs. Expense less D&A is explicitly labelled as an approximation when used.
- ROIC and incremental return on capital: require tax/pretax/EBIT and current/prior-year debt, equity and cash to derive NOPAT and average invested capital.
- EV and EV/Sales: strict calculation requires market cap, debt, preferred equity, non-controlling interests and cash; missing components are never assumed zero.
- Tangible Book Value/share: requires equity, goodwill, other intangibles and shares.
- Dividend/shareholder yield metrics: require actual dividend events or matching cash-flow buyback/issuance fields.
- DSCR: requires disclosed cash available for debt service, principal repayments and interest due.
- Altman Z: N/A unless company type/model and market equity at the same FY-end are available; current market cap is not silently mixed with an older balance sheet.
- Piotroski F-Score: N/A unless every one of nine tests can be evaluated using consecutive-year inputs, including a share-count dilution proxy.
- Bank/NBFC/insurance/hospital/IT/telecom/metal ratios: N/A where a company is eligible but provider fields are absent; marked Not applicable — sector when the selected company is outside the ratio's sector scope.

## Test report

Command: `python -m unittest discover -s tests -v`

- **30 tests passed**.
- **0 tests failed** in the final run.
- Ratio tests cover EBITDA margin, cash ratio, working capital, working-capital-to-sales, ROIC, incremental ROC, FCF margin, gross-profit growth, EPS growth, inventory turnover, Book Value/share, dividend growth, PEG unit convention, missing credit-sales behavior, zero denominators, cash-only requirement, strict EV component requirements, complete Piotroski input gating, Altman period matching, sector eligibility, empty data and duplicate prevention.
- Market tests cover 1D/5D price-return maths, insufficient history, zero base prices, missing closes, period filter list and sector-index map integrity.

## What these tests do not prove

The unit fixtures use synthetic numbers solely to validate formulas; they are not used as live app data. I have not claimed a company-by-company reconciliation to audited filings in this environment. Financial figures still come from InvestorLens's existing Yahoo Finance/yfinance adapter and its optional quote fallback. It may be incomplete, delayed, rate-limited or different from official consolidated/standalone filings. Validate target companies against NSE/company annual reports before investment reliance.

## Automatic market update caveat

No WebSocket subscription is exposed by the current yfinance adapter. The new tab therefore uses polling at a conservative default 300-second interval, configurable with `MARKET_POLL_SECONDS`, and honors available provider data. This is not a guarantee of real-time exchange prices; an authorized/licensed vendor or broker WebSocket/API is needed for contractual real-time NSE data and intraday cumulative volume. NIFTY industry charts use actual returned history; unsupported index symbols are omitted rather than assigned invented returns.
