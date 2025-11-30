# mortgage_app_v2.py - Integrated with Advanced ChatGPT Logic
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ---------------------------
# Reset session state so every new visitor gets fresh fields
# ---------------------------
if "initialized" not in st.session_state:
    st.session_state.clear()
    st.session_state["lump_events"] = []  # list of {"amount": float, "month": int}
    st.session_state["initialized"] = True

st.set_page_config(page_title="Mortgage vs Invest Planner (2025) — Advanced PV/Tax Model", layout="wide", page_icon="🏦")

# ---------------------------
# Helpers / formatters / io
# ---------------------------
def fmt_usd(x):
    try:
        return f"${float(x):,.2f}"
    except:
        return str(x)

def fmt_pct(x):
    try:
        return f"{x:.2f}%"
    except:
        return str(x)

def to_excel_bytes(df, sheet_name="Sheet1"):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return out.getvalue()

# ---------------------------
# Constants: MID cap (loan orig >= Dec 16, 2017) -> $750,000 for joint filers
# ---------------------------
MID_CAP_DEFAULT = 750000.0

# ---------------------------
# 2025 standard deductions (from ChatGPT source)
# ---------------------------
STANDARD_DEDUCTION_2025 = {
    "Single": 15750.0,
    "Married Filing Separately": 15750.0,
    "Head of Household": 23625.0,
    "Married Filing Jointly": 31500.0,
    "Surviving Spouses": 31500.0,
}

# ---------------------------
# 2025 federal tax brackets (cutpoints from ChatGPT source)
# ---------------------------
FEDERAL_BRACKETS_2025 = {
    "Single": [
        (0, 11925, 0.10), (11925, 48475, 0.12), (48475, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250525, 0.32), (250525, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Married Filing Jointly": [
        (0, 23850, 0.10), (23850, 96950, 0.12), (96950, 206700, 0.22),
        (206700, 394600, 0.24), (394600, 501050, 0.32), (501050, 751600, 0.35),
        (751600, float("inf"), 0.37),
    ],
    "Married Filing Separately": [
        (0, 11925, 0.10), (11925, 48475, 0.12), (48475, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250525, 0.32), (250525, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Head of Household": [
        (0, 17000, 0.10), (17000, 64850, 0.12), (64850, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250500, 0.32), (250500, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Surviving Spouses": [
        (0, 23850, 0.10), (23850, 96950, 0.12), (96950, 206700, 0.22),
        (206700, 394600, 0.24), (394600, 501050, 0.32), (501050, 751600, 0.35),
        (751600, float("inf"), 0.37),
    ],
}

# ---------------------------
# Tax engine: progressive federal tax (from ChatGPT source)
# ---------------------------
def compute_progressive_tax(taxable_income, filing_status):
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    # Use the progressive tax brackets to calculate tax owed
    for low, high, rate in FEDERAL_BRACKETS_2025[filing_status]:
        if taxable_income > low:
            taxed = min(taxable_income, high) - low
            if taxed > 0:
                tax += taxed * rate
        else:
            break
    return tax

# ---------------------------
# Mortgage math & amortization (all functions from ChatGPT source)
# ---------------------------
def compute_emi(P, annual_rate_pct, n_months):
    r = annual_rate_pct / 100.0 / 12.0
    if n_months <= 0: return 0.0
    if r == 0: return P / n_months
    powv = (1 + r) ** n_months
    return P * r * powv / (powv - 1)

def amortization_schedule(P, annual_rate_pct, n_months, extra_monthly=0.0):
    monthly_r = annual_rate_pct / 100.0 / 12.0
    base_emi = compute_emi(P, annual_rate_pct, n_months)
    balance = float(P)
    month = 0
    rows = []
    cap = max(n_months * 10, 1200)
    while balance > 0.005 and month < cap:
        month += 1
        interest = balance * monthly_r
        principal_base = base_emi - interest
        if principal_base < 0: principal_base = 0.0
        extra = min(extra_monthly, max(0.0, balance - principal_base))
        
        # Check for final payment
        if principal_base + extra >= balance - 1e-9:
            principal_paid = balance
            payment = interest + principal_paid
            balance = 0.0
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
            break
        else:
            principal_paid = principal_base + extra
            balance -= principal_paid
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(base_emi + extra,2), round(balance,2)])
    return pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])

# Apply multiple lump sums (apply AFTER that month's EMI) and re-simulate
def apply_multiple_lumps_and_resimulate(P, annual_rate_pct, n_months, lumps, extra_monthly=0.0):
    if not lumps:
        return amortization_schedule(P, annual_rate_pct, n_months, extra_monthly=extra_monthly)
    lumps_sorted = sorted(lumps, key=lambda x: int(x["month"]))
    monthly_r = annual_rate_pct / 100.0 / 12.0
    base_emi = compute_emi(P, annual_rate_pct, n_months)
    balance = float(P)
    month = 0
    rows = []
    cap = max(n_months * 10, 1200)
    lump_idx = 0
    while balance > 0.005 and month < cap:
        month += 1
        interest = balance * monthly_r
        principal_base = base_emi - interest
        if principal_base < 0: principal_base = 0.0
        extra = 0.0
        
        # Regular payment calculation and application
        if principal_base >= balance:
            principal_paid = balance
            payment = interest + principal_paid
            balance = 0.0
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
            break
        else:
            payment = base_emi
            balance -= principal_base
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(payment,2), round(balance,2)])
        
        # Apply lumps scheduled for this month AFTER the regular payment
        while lump_idx < len(lumps_sorted) and int(lumps_sorted[lump_idx]["month"]) == month:
            applied = min(float(lumps_sorted[lump_idx]["amount"]), balance)
            balance -= applied
            lump_idx += 1
            if balance <= 0.005: # Loan paid off by lump
                # Add one final entry to show zero balance
                rows.append([month, 0.0, 0.0, round(0.0,2), round(0.0,2), round(0.0,2), round(balance,2)])
                break
        if balance <= 0.005: break

    # Continue amortization with extra_monthly if still outstanding (after lumps)
    if balance > 0.005 and month < cap:
        while balance > 0.005 and month < cap:
            month += 1
            interest = balance * monthly_r
            principal_base = base_emi - interest
            if principal_base < 0: principal_base = 0.0
            extra = min(extra_monthly, max(0.0, balance - principal_base)) if extra_monthly > 0 else 0.0
            if principal_base + extra >= balance - 1e-9:
                principal_paid = balance
                payment = interest + principal_paid
                balance = 0.0
                rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
                break
            else:
                principal_paid = principal_base + extra
                balance -= principal_paid
                rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(base_emi + extra,2), round(balance,2)])
    return pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])

# ---------------------------
# Investment simulation: multiple lumps + monthly invest (from ChatGPT source)
# ---------------------------
def simulate_investment_multiple_lumps(lump_events, monthly_invest, annual_return_pct, invest_months):
    r_month = (1 + annual_return_pct/100.0) ** (1/12.0) - 1.0
    balance = 0.0
    # map lumps by month for quick add
    lumps_by_month = {}
    for e in lump_events:
        m = int(e["month"])
        lumps_by_month.setdefault(m, 0.0)
        lumps_by_month[m] += float(e["amount"])
        
    start_monthly = min([int(e["month"]) for e in lump_events]) if lump_events else 1
    
    for m in range(1, invest_months + 1):
        balance = balance * (1 + r_month)
        if m in lumps_by_month and m <= invest_months:
            balance += lumps_by_month[m]
        # Monthly invest starts the same month as the first lump (or month 1 if no lumps)
        if monthly_invest > 0 and m >= start_monthly:
            balance += monthly_invest
    return balance

# ---------------------------
# Tax savings calculator (based on ChatGPT source, simplified state tax logic removed)
# ---------------------------
def compute_annual_tax_savings(first_year_interest, income, filing_status, num_dependents, annual_property_tax, salt_cap=10000.0):
    std = STANDARD_DEDUCTION_2025.get(filing_status, 0.0)
    salt = min(annual_property_tax, salt_cap)
    itemized = first_year_interest + salt
    
    taxable_standard = max(0.0, income - std)
    taxable_itemized = max(0.0, income - itemized)
    
    tax_standard = compute_progressive_tax(taxable_standard, filing_status)
    tax_itemized = compute_progressive_tax(taxable_itemized, filing_status)
    
    CHILD_TAX_CREDIT = 2000.0
    credit = num_dependents * CHILD_TAX_CREDIT
    
    # Tax liability after standard deduction and credit
    tax_after_standard = max(0.0, tax_standard - credit)
    # Tax liability after itemized deduction and credit
    tax_after_itemized = max(0.0, tax_itemized - credit)
    
    federal_savings = tax_after_standard - tax_after_itemized
    
    # Total tax savings is the reduction in tax liability by itemizing
    total = max(0.0, federal_savings) 
    return total

# Compute average starting balance for MID cap logic (from ChatGPT source)
def compute_first_year_avg_start_balance(df_sched, starting_principal):
    """Approximate average starting balance for months 1..12."""
    starts = []
    for m in range(1, 13):
        if m == 1:
            starts.append(starting_principal)
        else:
            prev = df_sched.loc[df_sched["Month"] == (m - 1)]
            if not prev.empty:
                # Start of month m is end balance of month m-1
                starts.append(float(prev["Balance"].values[0]))
            else:
                starts.append(0.0)
    arr = np.array(starts)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


# ---------------------------
# UI - Layout and Inputs (Adapted from ChatGPT source)
# ---------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("## 🏡 Mortgage & Financial Inputs")
    home_price = st.number_input("Home price ($)", value=770000.0, step=1000.0, format="%.2f")
    down_payment = st.number_input("Down payment ($)", value=0.0, step=1000.0, format="%.2f")
    remaining_loan = max(0.0, home_price - down_payment)
    annual_interest = st.number_input("Annual interest rate (%)", value=5.50, step=0.01, format="%.2f")
    remaining_years = st.number_input("Remaining tenure (years)", value=20, min_value=0, step=1)
    extra_monthly = st.number_input("Extra monthly payment (optional $)", value=0.0, step=50.0, format="%.2f")
    property_tax_rate = st.number_input("Property tax rate (%)", value=1.50, step=0.01, format="%.2f") / 100.0
    
    st.markdown("---")
    st.markdown("## 🇺🇸 Tax Inputs (US, 2025 Model)")
    filing_status = st.selectbox("Filing status (2025)", list(STANDARD_DEDUCTION_2025.keys()), index=3)
    income = st.number_input("Annual gross income ($)", value=150000.0, step=1000.0, format="%.2f")
    num_dependents = st.number_input("Number of dependents (children)", value=0, min_value=0, step=1)
    # Simplified tax logic, removed state tax rate input as it's not crucial for the comparison
    salt_cap = st.number_input("SALT cap ($)", value=10000.0, step=100.0, format="%.2f")
    
    st.markdown("---")
    st.markdown("## 📈 Investment Inputs (Multiple Lumps Supported)")
    new_lump_amount = st.number_input("New lump amount ($)", value=10000.0, step=100.0, format="%.2f", key="new_lump_amount")
    new_lump_month = st.number_input("New lump month (1 = now)", value=3, min_value=1, step=1, key="new_lump_month")
    
    col_add1, col_add2 = st.columns([1,1])
    with col_add1:
        if st.button("Add lump-sum"):
            st.session_state["lump_events"].append({"amount": float(new_lump_amount), "month": int(new_lump_month)})
    with col_add2:
        if st.button("Remove last lump"):
            if st.session_state["lump_events"]:
                st.session_state["lump_events"].pop()
    if st.button("Remove all lumps"):
        st.session_state["lump_events"] = []

    # Show current lumps
    st.write("### Current Lump-Sum Events:")
    if st.session_state["lump_events"]:
        df_lumps_display = pd.DataFrame(st.session_state["lump_events"])
        df_lumps_display.index = np.arange(1, len(df_lumps_display) + 1)
        st.dataframe(df_lumps_display.rename(columns={"amount":"Amount ($)", "month":"Month"}), use_container_width=True)
    else:
        st.write("_No lump-sum events added._")

    st.markdown("---")
    monthly_invest = st.number_input("Monthly invest amount (optional $)", value=0.0, step=10.0, format="%.2f")
    annual_return = st.number_input("Expected annual return (%)", value=5.0, step=0.1, format="%.2f")
    invest_horizon_years = st.number_input("Investment horizon (years)", value=10, min_value=1, step=1)

with right_col:
    st.title("Mortgage Prepay vs. Invest Analysis")
    
    if remaining_loan <= 0 or remaining_years <= 0 or annual_interest <= 0:
        st.error("Please enter valid positive values for Loan Amount, Rate, and Term.")
        st.stop()
    
    months = int(max(1, round(remaining_years * 12)))
    lumps = st.session_state["lump_events"]
    
    # 1. Amortization Schedules
    df_base = amortization_schedule(remaining_loan, annual_interest, months, extra_monthly=0.0)
    df_lump = apply_multiple_lumps_and_resimulate(remaining_loan, annual_interest, months, lumps, extra_monthly=extra_monthly)

    # 2. Totals & Savings
    total_interest_base = float(df_base["Interest"].sum()) if not df_base.empty else 0.0
    total_interest_lump = float(df_lump["Interest"].sum()) if not df_lump.empty else 0.0
    interest_saved_by_lump = max(0.0, total_interest_base - total_interest_lump)
    
    first_year_interest_base = float(df_base.loc[df_base["Month"] <= 12, "Interest"].sum()) if not df_base.empty else 0.0
    first_year_interest_lump = float(df_lump.loc[df_lump["Month"] <= 12, "Interest"].sum()) if not df_lump.empty else 0.0

    months_base = len(df_base)
    months_lump = len(df_lump)
    months_saved_by_lump = max(0, months_base - months_lump)

    # 3. Tax Calculations (MID Cap & Savings)
    annual_property_tax = home_price * property_tax_rate
    
    # MID Cap Prorating Logic
    avg_balance_base = compute_first_year_avg_start_balance(df_base, remaining_loan)
    avg_balance_lump = compute_first_year_avg_start_balance(df_lump, remaining_loan)

    cap = MID_CAP_DEFAULT
    cap_factor_base = min(1.0, cap / avg_balance_base) if avg_balance_base > 0 else 1.0
    cap_factor_lump = min(1.0, cap / avg_balance_lump) if avg_balance_lump > 0 else 1.0
    cap_applied_base = cap_factor_base < 1.0
    cap_applied_lump = cap_factor_lump < 1.0

    deductible_first_year_interest_base = first_year_interest_base * cap_factor_base
    deductible_first_year_interest_lump = first_year_interest_lump * cap_factor_lump

    # Annual Tax Savings using Deductible Interest
    tax_savings_base = compute_annual_tax_savings(deductible_first_year_interest_base, income, filing_status, num_dependents, annual_property_tax, salt_cap)
    tax_savings_lump = compute_annual_tax_savings(deductible_first_year_interest_lump, income, filing_status, num_dependents, annual_property_tax, salt_cap)

    lost_tax_savings_due_to_lumps = max(0.0, tax_savings_base - tax_savings_lump)

    # 4. Investment FV Simulation
    invest_months = int(invest_horizon_years * 12)
    annual_return = st.session_state.get("expected_annual_return", annual_return) # Use existing value if available
    monthly_return = (1 + annual_return / 100.0) ** (1/12.0) - 1.0
    inv_final_value = simulate_investment_multiple_lumps(lumps, monthly_invest, annual_return, invest_months)

    # 5. NPV Calculations (Core Comparison)
    max_months = max(len(df_base), len(df_lump))
    interest_base_series = np.zeros(max_months)
    interest_lump_series = np.zeros(max_months)

    for idx in range(max_months):
        m = idx + 1
        if m <= len(df_base):
            interest_base_series[idx] = float(df_base.loc[df_base["Month"] == m, "Interest"].values[0])
        if m <= len(df_lump):
            interest_lump_series[idx] = float(df_lump.loc[df_lump["Month"] == m, "Interest"].values[0])
            
    interest_saved_series = np.maximum(interest_base_series - interest_lump_series, 0.0)
    
    # Discount Factors for PV of monthly stream
    discount_factors = (1 + monthly_return) ** np.arange(1, max_months + 1)
    npv_mortgage_interest_savings = float(np.sum(interest_saved_series / discount_factors))

    # PV of lost tax savings: treated as annual amount at end of year 1 (month 12)
    pv_lost_tax_savings = 0.0
    if lost_tax_savings_due_to_lumps > 0:
        pv_lost_tax_savings = lost_tax_savings_due_to_lumps / ((1 + monthly_return) ** 12)
        
    # NET MORTGAGE NPV: Interest savings PV minus the value of lost tax breaks
    npv_mortgage_net = npv_mortgage_interest_savings - pv_lost_tax_savings
    
    # PV of investment FV
    pv_invest = inv_final_value / ((1 + monthly_return) ** invest_months) if invest_months > 0 else inv_final_value

    # ---------------------------
    # Summary Cards
    # ---------------------------
    st.subheader("Quick Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Loan Principal", fmt_usd(remaining_loan))
        st.write("Annual Interest Rate:", fmt_pct(annual_interest))
    with c2:
        st.metric("Base EMI", fmt_usd(compute_emi(remaining_loan, annual_interest, months)))
        st.write("Monthly Property Tax (est.):", fmt_usd(annual_property_tax / 12.0))
    with c3:
        st.metric("Total Interest Saved (Nominal)", fmt_usd(interest_saved_by_lump))
        st.write("Payoff Accelerated:", f"**{months_saved_by_lump} months**")
    with c4:
        st.metric("Investment FV (after horizon)", fmt_usd(inv_final_value))
        st.write("Investment Horizon:", f"**{invest_horizon_years} years**")

    st.markdown("---")

    # ---------------------------
    # Comparison (PV is the key figure)
    # ---------------------------
    st.subheader("🏦 Lump-sum Prepay vs Invest — Net Present Value (NPV) Comparison")

    left, right = st.columns(2)
    with left:
        st.markdown("### Option A: Prepay Mortgage (Net NPV)")
        st.write(f"NPV of monthly interest saved: **{fmt_usd(npv_mortgage_interest_savings)}**")
        st.write(f"PV of lost tax savings (approx): **{fmt_usd(pv_lost_tax_savings)}**")
        st.markdown(f"**Net Mortgage NPV:** **{fmt_usd(npv_mortgage_net)}**")

    with right:
        st.markdown("### Option B: Invest Instead")
        st.write(f"Future Value (FV): **{fmt_usd(inv_final_value)}**")
        st.write(f"Discount Rate Used (Annual): **{fmt_pct(annual_return)}**")
        st.markdown(f"**NPV of Investment Future Value:** **{fmt_usd(pv_invest)}**")

    st.markdown("---")
    st.subheader("🎯 Recommendation")

    # PV recommendation (apples-to-apples)
    if pv_invest > npv_mortgage_net:
        st.success(f"**INVESTING** outperforms prepaying by approx **{fmt_usd(pv_invest - npv_mortgage_net)}** in present value terms.")
    else:
        st.info(f"**PREPAYING** outperforms investing by approx **{fmt_usd(npv_mortgage_net - pv_invest)}** in present value terms.")
        
    st.markdown("---")
    
    # ---------------------------
    # Detailed Tax Info
    # ---------------------------
    st.subheader("Tax Details (First Year Estimates) 🧾")
    st.write(f"Filing status: **{filing_status}** | Annual income: **{fmt_usd(income)}**")
    
    tax_col1, tax_col2 = st.columns(2)
    with tax_col1:
        st.markdown("**Base Mortgage Scenario**")
        st.write(f"First-Year Interest: **{fmt_usd(first_year_interest_base)}**")
        if cap_applied_base:
            st.write(f"*MID Cap ($\${int(MID_CAP_DEFAULT):,}$) applied.* Deductible Interest Used: **{fmt_usd(deductible_first_year_interest_base)}**")
        st.write(f"Annual Tax Savings (Est.): **{fmt_usd(tax_savings_base)}**")

    with tax_col2:
        st.markdown("**Prepay Scenario**")
        st.write(f"First-Year Interest: **{fmt_usd(first_year_interest_lump)}**")
        if cap_applied_lump:
            st.write(f"*MID Cap ($\${int(MID_CAP_DEFAULT):,}$) applied.* Deductible Interest Used: **{fmt_usd(deductible_first_year_interest_lump)}**")
        st.write(f"Annual Tax Savings (Est.): **{fmt_usd(tax_savings_lump)}**")
        st.write(f"**Lost Tax Savings by Prepaying (Nominal):** **{fmt_usd(lost_tax_savings_due_to_lumps)}**")
    
    st.caption("Tax estimates use 2025 federal brackets and rules, including the $10,000 SALT cap and the $750,000 Mortgage Interest Deduction (MID) cap. Consult a tax professional.")
