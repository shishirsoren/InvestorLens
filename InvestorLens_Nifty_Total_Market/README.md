# InvestorLens — Audited 5-FY Build

This build fixes the financial-statement display layer so annual periods are represented as Indian financial years (April–March) and statement tables use the correct row/column orientation.

## What was added/fixed
- 5 financial-year presentation for Income Statement, Balance Sheet and Cash Flow where provider history permits.
- Indian FY labels such as FY2021-22, FY2022-23, FY2023-24, FY2024-25 and FY2025-26.
- Quarterly fallback aggregation uses Apr–Mar fiscal years: flow items are summed; balance-sheet items use the latest period in the FY.
- Income Statement and Cash Flow tables now correctly read statement line items from columns.
- Historical ROE/ROA use average beginning/ending equity/assets consistently.
- Added a Data Audit tab showing provider, period coverage, units and completeness status.
- The app does not invent missing years; if the provider returns fewer than five annual/fiscal periods, it clearly reports INCOMPLETE.

## Important data note
The current data adapter still uses Yahoo Finance/yfinance for company fundamentals. Before public investment use, reconcile major figures against official NSE/company filings and consider a licensed/authoritative fundamentals feed for the full 1,000-company universe.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
