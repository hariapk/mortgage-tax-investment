# CLAUDE.md — Mortgage & Tax Investment Planner

## Project Overview

A Streamlit web app for US homeowners to analyze mortgage decisions vs. investment alternatives. It models:
- Mortgage amortization (base, extra monthly payments, lump-sum prepayment)
- 2025 US federal tax deductions (mortgage interest, SALT, child tax credit)
- Lump-sum prepay vs. invest NPV comparison

**Main file:** `mortgage_tax_investment_app.py`

## Running the App

```bash
pip install streamlit pandas numpy openpyxl altair
streamlit run mortgage_tax_investment_app.py
```

## Architecture

The file is a single-module Streamlit app with three layers:

### 1. Math/Finance Core (pure functions, no UI)

| Function | Purpose |
|---|---|
| `compute_emi(P, rate, n_months)` | Monthly payment using standard amortization formula |
| `amortization_schedule(P, rate, n_months, extra_monthly)` | Full month-by-month schedule until payoff |
| `apply_lump_and_resimulate(...)` | Applies a lump sum at a specified month, resimulates remaining schedule |
| `simulate_investment(lump, monthly, return_pct, months)` | Compounds a lump + monthly deposits using monthly compounding |
| `compute_progressive_tax(taxable_income, filing_status)` | Progressive 2025 federal brackets |
| `compute_annual_tax_savings(...)` | Federal + state tax savings from itemizing mortgage interest + SALT |
| `calculate_npv(...)` | NPV of prepay vs. invest: discounts interest savings and FV of investment |

### 2. Tax Data Constants

- `STANDARD_DEDUCTION_2025` — by filing status
- `FEDERAL_BRACKETS_2025` — 2025 brackets for all filing statuses
- Child tax credit hardcoded at `$2,000/dependent`
- SALT cap defaults to `$10,000`

### 3. Streamlit UI

- **Sidebar:** All inputs (mortgage, tax, investment) — dark blue gradient sidebar
- **Main area (full width):** All outputs (metrics, charts, tables, downloads)

### 4. UI Helper

| Function | Purpose |
|---|---|
| `npv_card_html(icon, title, npv_val, detail_html, accent)` | Returns self-contained inline-styled HTML for NPV comparison cards |
| `cumsum_padded(df, max_len)` | Returns cumulative interest array padded to `max_len` (fixes length mismatch when schedules differ in length) |

## Key Design Decisions

### NPV Methodology (`calculate_npv`)
- Discount rate = monthly investment return rate (not mortgage rate)
- **Option A (Prepay):** NPV of month-by-month interest savings, each discounted at `(1+r)^(t+1)`
- **Option B (Invest):** FV of lump invested for `(horizon - delay)` months, discounted back to month 0
- `lump_month=1` means no delay (invest immediately); `lump_month=12` means 11 months of delay
- Delayed investment computes `compounding_months = invest_horizon_months - (lump_month - 1)`

### Tax Savings Calculation
- Compares federal tax under standard deduction vs. itemized (mortgage interest + SALT)
- State savings approximated as `(itemized - std_deduction) * state_rate` — not actual state brackets
- Tax savings only materialize if itemized deduction exceeds standard deduction

### Amortization with Lump Sum
- `apply_lump_and_resimulate` keeps the original EMI constant after the lump (does not recast)
- Lump is applied immediately after the scheduled payment at `lump_month`
- After lump, `extra_monthly` resumes if provided

## Input Defaults (for reference)

| Input | Default |
|---|---|
| Home price | $500,000 |
| Down payment | $100,000 |
| Annual interest rate | 6.5% |
| Tenure | 20 years |
| Filing status | Married Filing Jointly |
| Annual income | $150,000 |
| Lump-sum amount | $20,000 |
| Lump-sum month | 12 |
| Expected annual return | 10% |

## Output Sections (main area)

1. **Hero Banner** — Gradient header showing loan summary at a glance
2. **Quick Summary** — 4 native `st.metric` cards: EMI, first-year interest, tax savings, effective rate
3. **NPV Comparison** — Two side-by-side inline-styled cards (orange=prepay, green=invest) + native `st.success/info/warning` recommendation
4. **Balance & Cumulative Interest Charts** — Side-by-side Altair line charts comparing 3 scenarios (blue=base, green=extra, orange=lump)
5. **Amortization Tables** — Tabs for base / extra / lump, with CSV and Excel downloads
6. **Yearly Summary** — Aggregated by year for base schedule
7. **Tax Details** — Itemized key/value rows with inline styles

## UI Approach & Gotchas

- **Do NOT use `st.markdown('<div class="X">')` + native widgets + `st.markdown('</div>')`** — Streamlit renders each `st.markdown` call in its own container; you cannot open/close a wrapper div across multiple calls.
- **Do NOT use CSS class-based layouts for metric cards** — `st.markdown` with `unsafe_allow_html=True` does not reliably render complex nested `<div>` grids. Use `st.columns` + `st.metric` instead.
- **Self-contained HTML only** — Any custom HTML passed to `st.markdown` must be a single complete block with inline styles. CSS classes only work reliably for simple elements (like the hero banner).
- **Chart length mismatch** — When schedules have different lengths, use `cumsum_padded()` to align arrays to `max_m` before assigning to `df_plot`.

## Dependencies

```
streamlit
pandas
numpy
openpyxl      # Excel export
altair        # Charts
```

## Known Limitations / Notes

- State tax savings use a flat-rate approximation, not actual state brackets
- `$2,000` child tax credit is hardcoded (simplified; does not phase out at higher incomes)
- Investment simulation assumes all returns are realized (no tax on investment gains modeled)
- NPV recommendation ignores risk premium — investing at 10% is not equivalent to guaranteed mortgage interest savings
- For precise tax planning, consult a tax professional (this is explicitly noted in the UI)
