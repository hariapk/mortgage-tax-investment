# Financial Accuracy Rules

- Always use monthly compounding: `(1 + annual_rate/100/12)` — never daily or continuous
- EMI formula: `P * r * (1+r)^n / ((1+r)^n - 1)` where r = monthly rate, n = months
- NPV discount rate = monthly investment return rate (not mortgage rate)
- Lump-sum prepay does NOT recast the loan — original EMI stays constant after lump
- Tax savings only apply when itemized deductions exceed the standard deduction
- State tax savings use a flat-rate approximation — not actual state brackets (document this in UI)
- SALT cap defaults to $10,000; child tax credit hardcoded at $2,000/dependent
- All monetary outputs should be rounded to 2 decimal places for display
