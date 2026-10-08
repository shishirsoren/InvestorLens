# InvestorLens v4 — 5-FY Financial Statement Fix

## Fixed
1. Financial statement tables previously expected statement line items in the DataFrame index while the normalized annual data stored them in columns. This caused missing/empty Income Statement and Cash Flow displays.
2. Annual periods are now labeled using India's April–March financial year.
3. Quarterly fallback is aggregated by Indian FY rather than calendar year.
4. Flow items are summed; balance-sheet items use the latest period in each FY.
5. Historical ROE and ROA now use average beginning/ending equity/assets rather than only ending balances.
6. Added a Data Audit tab for statement-period coverage and source/units.

## Limitation
Yahoo Finance/yfinance may still provide fewer than five historical annual/quarterly statement periods for some companies. The application reports this rather than fabricating a fifth year. For an investor-grade public deployment, an authoritative NSE/company/XBRL fundamentals pipeline or licensed fundamentals provider should be added.
