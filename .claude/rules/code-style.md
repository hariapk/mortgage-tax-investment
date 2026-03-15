# Code Style — Mortgage Planner

- Use descriptive variable names: `monthly_rate` not `r`, `loan_balance` not `b`
- Pure finance/math functions must have no Streamlit imports or UI side effects
- Keep all pure functions above the Streamlit UI section in `mortgage_tax_investment_app.py`
- Use f-strings for formatting; avoid `%` or `.format()`
- Constants (tax brackets, standard deductions) go in the Tax Data Constants section, not inline
- Do not add docstrings or comments to code that wasn't changed
