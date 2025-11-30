# mortgage_tax_investment_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
import altair as alt

# ---------------------------
# Formatting helpers
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
# 2025 standard deductions (final)
# ---------------------------
STANDARD_DEDUCTION_2025 = {
    "Single": 15750.0,
    "Married Filing Separately": 15750.0,
    "Head of Household": 23625.0,
    "Married Filing Jointly": 31500.0,
    "Surviving Spouses": 31500.0,
}

# ---------------------------
# 2025 federal tax brackets (cutpoints)
# ---------------------------
FEDERAL_BRACKETS_2025 = {
    "Single": [
        (0, 11925, 0.10),
        (11925, 48475, 0.12),
        (48475, 103350, 0.22),
        (103350, 197300, 0.24),
        (197300, 250525, 0.32),
        (250525, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Married Filing Jointly": [
        (0, 23850, 0.10),
        (23850, 96950, 0.12),
        (96950, 206700, 0.22),
        (206700, 394600, 0.24),
        (394600, 501050, 0.32),
        (501050, 751600, 0.35),
        (751600, float("inf"), 0.37),
    ],
    "Married Filing Separately": [
        (0, 11925, 0.10),
        (11925, 48475, 0.12),
        (48475, 103350, 0.22),
        (103350, 197300, 0.24),
        (197300, 250525, 0.32),
        (250525, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Head of Household": [
        (0, 17000, 0.10),
        (17000, 64850, 0.12),
        (64850, 103350, 0.22),
        (103350, 197300, 0.24),
        (197300, 250500, 0.32),
        (250500, 626350, 0.35),
        (626350, float("inf"), 0.37),
    ],
    "Surviving Spouses": [
        (0, 23850, 0.10),
        (23850, 96950, 0.12),
        (96950, 206700, 0.22),
        (206700, 394600, 0.24),
        (394600, 501050, 0.32),
        (501050, 751600, 0.35),
        (751600, float("inf"), 0.37),
    ],
}

# ---------------------------
# Tax engine: progressive federal tax
# ---------------------------
def compute_progressive_tax(taxable_income, filing_status):
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    for low, high, rate in FEDERAL_BRACKETS_2025[filing_status]:
        if taxable_income > low:
            taxed = min(taxable_income, high) - low
            if taxed > 0:
                tax += taxed * rate
        else:
            break
    return tax

# ---------------------------
# Mortgage math & amortization
# ---------------------------
def compute_emi(P, annual_rate_pct, n_months):
    r = annual_rate_pct / 100.0 / 12.0
    if n_months <= 0:
        return 0.0
    if r == 0:
        return P / n_months
    powv = (1 + r) ** n_months
    return P * r * powv / (powv - 1)

def amortization_schedule(P, annual_rate_pct, n_months, extra_monthly=0.0):
    """
    Builds an amortization schedule until payoff.
    Columns: Month, BaseEMI, Interest, PrincipalBase, ExtraPayment, TotalPayment, Balance
    """
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
        if principal_base < 0:
            principal_base = 0.0
        extra = min(extra_monthly, max(0.0, balance - principal_base))
        payment = base_emi + extra
        if principal_base + extra >= balance - 1e-9:
            # final payment
            principal_paid = balance
            payment = interest + principal_paid
            balance = 0.0
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
            break
        else:
            principal_paid = principal_base + extra
            balance -= principal_paid
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(payment,2), round(balance,2)])
    df = pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])
    return df

# ---------------------------
# Lump-sum simulation: keep EMI constant, apply lump at month t, recompute schedule
# ---------------------------
def apply_lump_and_resimulate(original_schedule_df, P, annual_rate_pct, n_months, lump_amount, lump_month, extra_monthly=0.0):
    """
    original_schedule_df: schedule without lump (so we can use months)
    P, annual_rate_pct, n_months: original loan parameters
    lump_amount: amount applied to principal at lump_month
    lump_month: integer >=1 (1 = immediate today/month1)
    Returns: new_schedule_df, months_new
    """
    # Simulate month-by-month: build schedule up to lump_month, then subtract lump from balance and continue amortization
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
        if principal_base < 0:
            principal_base = 0.0
        extra = 0.0 # we treat extra_monthly separately only after lump if provided
        # normal monthly payment before lump:
        if principal_base >= balance:
            # final payment without lump
            principal_paid = balance
            payment = interest + principal_paid
            balance = 0.0
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
            break
        else:
            payment = base_emi
            balance = balance - principal_base
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(payment,2), round(balance,2)])
        # If this is the lump month, apply lump immediately after that month's scheduled payment
        if month == lump_month:
            applied = min(lump_amount, balance)
            balance = balance - applied
            # if applying lump pays off the loan immediately:
            if balance <= 0.005:
                # finalizing: create a final row capturing this (we'll show an extra row marking post-lump payoff)
                month += 1
                # no interest accrues for the next month
                rows.append([month, 0.0, 0.0, round(0.0,2), round(0.0,2), round(0.0,2), round(0.0,2)])
                break
            # after applying lump, continue the loop which will create further months until payoff using same base_emi but reduced balance
    # If loan not paid yet, continue with same base_emi and include extra_monthly if provided (we assume extra recurring applies after lump, if user wants that)
    # But we built rows only up to the lump month and possibly not beyond; rebuild final schedule starting from current balance and month count
    if balance > 0.005 and month < cap:
        # continue amortization from current balance, but allow recurring extra_monthly thereafter
        while balance > 0.005 and month < cap:
            month += 1
            interest = balance * monthly_r
            principal_base = base_emi - interest
            if principal_base < 0:
                principal_base = 0.0
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
    df_new = pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])
    return df_new

# ---------------------------
# Investment simulation (monthly compounding)
# ---------------------------
def simulate_investment(lump_amount, monthly_invest=0.0, annual_return_pct=10.0, months=12):
    """
    Simulate starting lump_amount invested at month 0, plus monthly_invest deposits at the END of each month.
    monthly compounding used: monthly_return = (1+annual_return)^(1/12)-1
    Return final balance after 'months'.
    """
    r_month = (1 + annual_return_pct/100.0) ** (1/12.0) - 1.0
    balance = float(lump_amount)
    for m in range(1, months+1):
        # grow existing balance
        balance = balance * (1 + r_month)
        # then add monthly contribution at month end
        if monthly_invest > 0:
            balance += monthly_invest
    return balance

# ---------------------------
# Tax savings calculator (uses first-year interest as deduction proxy)
# ---------------------------
def compute_annual_tax_savings(first_year_interest, income, filing_status, num_dependents, include_state, state_tax_rate, annual_property_tax, salt_cap=10000.0):
    """
    Returns:
      annual_tax_savings (float),
      breakdown: (federal_savings, state_savings)
    Approach:
      - itemized = first_year_interest + min(annual_property_tax, salt_cap)
      - compare federal tax liability under standard deduction vs itemized
    """
    std = STANDARD_DEDUCTION_2025.get(filing_status, 0.0)
    salt = min(annual_property_tax, salt_cap)
    itemized = first_year_interest + salt
    # taxable incomes
    taxable_standard = max(0.0, income - std)
    taxable_itemized = max(0.0, income - itemized)
    # federal taxes
    tax_standard = compute_progressive_tax(taxable_standard, filing_status)
    tax_itemized = compute_progressive_tax(taxable_itemized, filing_status)
    # child credit
    CHILD_TAX_CREDIT = 2000.0
    credit = num_dependents * CHILD_TAX_CREDIT
    tax_after_standard = max(0.0, tax_standard - credit)
    tax_after_itemized = max(0.0, tax_itemized - credit)
    federal_savings = tax_after_standard - tax_after_itemized
    # state approximate: assume state tax savings from deduction is: (itemized - std) * state_tax_rate
    state_savings = 0.0
    if include_state:
        # State deduction benefit is realized if itemized > standard
        state_deduction_benefit = max(0.0, itemized - std)
        state_savings = state_deduction_benefit * state_tax_rate
    
    total = max(0.0, federal_savings + state_savings)
    return total, federal_savings, state_savings

# ---------------------------
# NPV CALCULATION FUNCTION (FIXED)
# ---------------------------
def calculate_npv(df_base, df_scenario, lump_amount, monthly_invest, annual_r_invest, invest_horizon_months, interest_saved_total, tax_savings_annual_base, tax_savings_annual_scenario, months_lump):
    """
    Compares the NPV of the Lump Payment (Prepay) vs. Lump Investment (Invest) option.
    Discount Rate (r) used is the monthly investment return rate.

    NPV_A: Present value of total interest saved (discounted at investment rate)
    NPV_B: Present value of investment future value
    """
    
    # Calculate monthly investment discount rate (r)
    r_month = (1 + annual_r_invest / 100.0) ** (1/12.0) - 1.0
    
    # 1. Option A: Prepay Mortgage (Benefit = Interest Saved)
    max_m = max(len(df_base), len(df_scenario))
    
    # FIX: Set 'Month' as index first, then reindex and reset to avoid duplicate column error.
    
    # Prepare base DataFrame
    df_base_indexed = df_base.set_index('Month')
    # Reindex to max length and fill N/A with 0.0, then reset index to bring 'Month' back as a column
    df_base_padded = df_base_indexed.reindex(range(1, max_m + 1), fill_value=0.0).reset_index(names=['Month'])
    
    # Prepare scenario DataFrame
    df_scenario_indexed = df_scenario.set_index('Month')
    df_scenario_padded = df_scenario_indexed.reindex(range(1, max_m + 1), fill_value=0.0).reset_index(names=['Month'])
    
    # Calculate the net interest saved in each month (Base Interest - Scenario Interest)
    interest_saved_monthly = df_base_padded['Interest'] - df_scenario_padded['Interest']
    
    # Calculate NPV of the Interest Savings stream
    npv_prepay_interest = np.sum([
        interest_saved_monthly.iloc[t] / ((1 + r_month) ** (t + 1)) 
        for t in range(len(interest_saved_monthly))
    ])
    
    # 2. Option B: Invest Lump Sum (Cost = Lump Amount, Benefit = Investment Return)
    inv_final_value = simulate_investment(lump_amount, monthly_invest=monthly_invest, annual_return_pct=annual_r_invest, months=invest_horizon_months)
    
    # Calculate PV of the Investment Future Value
    # This PV represents the value of the investment choice today.
    npv_invest_value = inv_final_value / ((1 + r_month) ** invest_horizon_months)

    # Simplified NPV Comparison: PV of Interest Saved vs. PV of Investment FV
    return npv_prepay_interest, npv_invest_value

# ---------------------------
# UI: Left columns split into Mortgage / Tax / Investment inputs
# ---------------------------
st.set_page_config(page_title="Mortgage vs Invest Planner (2025)", layout="wide", page_icon="🏦")
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Mortgage inputs")
    home_price = st.number_input("Home price ($)", value=500000.0, step=1000.0, format="%.2f")
    down_payment = st.number_input("Down payment ($)", value=100000.0, step=1000.0, format="%.2f")
    remaining_loan = max(0.0, home_price - down_payment)
    annual_interest = st.number_input("Annual interest rate (%)", value=6.5, step=0.01, format="%.2f")
    remaining_years = st.number_input("Remaining tenure (years)", value=20, min_value=0, step=1)
    extra_monthly = st.number_input("Extra monthly payment (optional $)", value=0.0, step=50.0, format="%.2f")
    property_tax_rate = st.number_input("Property tax rate (%)", value=1.0, step=0.01, format="%.2f") / 100.0

    st.markdown("---")
    st.header("Tax inputs (US, 2025)")
    filing_status = st.selectbox("Filing status (2025)", list(STANDARD_DEDUCTION_2025.keys()), index=3)  # default Married Filing Jointly
    income = st.number_input("Annual gross income ($)", value=150000.0, step=1000.0, format="%.2f")
    num_dependents = st.number_input("Number of dependents (children)", value=0, min_value=0, step=1)
    include_state = st.checkbox("Include state tax in SALT modeling (approx)", value=False)
    state_tax_rate = st.number_input("State tax rate (%) (if enabled)", value=5.0, step=0.1, format="%.2f") / 100.0 if include_state else 0.0
    salt_cap = st.number_input("SALT cap ($)", value=10000.0, step=100.0, format="%.2f")

    st.markdown("---")
    st.header("Investment inputs")
    lump_amount = st.number_input("Lump-sum amount ($)", value=20000.0, step=100.0, format="%.2f")
    lump_month = st.number_input("Lump-sum month (1 = now)", value=12, min_value=1, step=1)
    monthly_invest = st.number_input("Monthly invest amount (optional $)", value=0.0, step=10.0, format="%.2f")
    annual_return = st.number_input("Expected annual return (%)", value=10.0, step=0.1, format="%.2f")  # default 10%
    invest_horizon_years = st.number_input("Investment horizon (years)", value=remaining_years, min_value=1, step=1)

with right_col:
    st.title("Mortgage — Analysis and Comparison")

    months = int(max(1, round(remaining_years * 12)))
    # Base schedule (no extra, no lump)
    df_base = amortization_schedule(remaining_loan, annual_interest, months, extra_monthly=0.0)
    # With extra monthly payments
    df_extra = amortization_schedule(remaining_loan, annual_interest, months, extra_monthly=extra_monthly if extra_monthly>0 else 0.0)
    # Lump-sum schedule
    df_lump = apply_lump_and_resimulate(df_base, remaining_loan, annual_interest, months, lump_amount, lump_month, extra_monthly=extra_monthly if extra_monthly>0 else 0.0)

    # Totals
    total_interest_base = df_base["Interest"].sum() if not df_base.empty else 0.0
    total_interest_lump = df_lump["Interest"].sum() if not df_lump.empty else 0.0

    # First-year interest (sum months 1..12)
    first_year_interest_base = df_base.loc[df_base["Month"] <= 12, "Interest"].sum() if not df_base.empty else 0.0
    first_year_interest_lump = df_lump.loc[df_lump["Month"] <= 12, "Interest"].sum() if not df_lump.empty else 0.0

    months_base = len(df_base)
    months_lump = len(df_lump)

    months_saved_by_lump = max(0, months_base - months_lump)
    interest_saved_by_lump = max(0.0, total_interest_base - total_interest_lump)

    # Tax calculations (annual tax savings from mortgage interest deduction)
    annual_property_tax = home_price * property_tax_rate
    # For base (original) tax savings
    tax_savings_base, fed_s_base, st_s_base = compute_annual_tax_savings(first_year_interest_base, income, filing_status, num_dependents, include_state, state_tax_rate, annual_property_tax, salt_cap)
    # For lump scenario, compute tax savings using lump schedule first-year interest
    tax_savings_lump, fed_s_lump, st_s_lump = compute_annual_tax_savings(first_year_interest_lump, income, filing_status, num_dependents, include_state, state_tax_rate, annual_property_tax, salt_cap)

    lost_tax_savings_due_to_lump = max(0.0, tax_savings_base - tax_savings_lump)

    # Investment simulation & NPV calculation
    invest_months = int(invest_horizon_years * 12)
    
    # Calculate NPVs
    npv_prepay, npv_invest = calculate_npv(
        df_base=df_base, 
        df_scenario=df_lump, 
        lump_amount=lump_amount, 
        monthly_invest=monthly_invest, 
        annual_r_invest=annual_return, 
        invest_horizon_months=invest_months,
        interest_saved_total=interest_saved_by_lump,
        tax_savings_annual_base=tax_savings_base,
        tax_savings_annual_scenario=tax_savings_lump,
        months_lump=months_lump
    )


    # Also compute effective APR after tax
    effective_first_year_interest_base = first_year_interest_base - tax_savings_base
    effective_rate_base_pct = (effective_first_year_interest_base / remaining_loan) * 100.0 if remaining_loan > 0 else 0.0

    effective_first_year_interest_lump = first_year_interest_lump - tax_savings_lump
    effective_rate_lump_pct = (effective_first_year_interest_lump / remaining_loan) * 100.0 if remaining_loan > 0 else 0.0

    # ---------------------------
    # Output summary cards
    # ---------------------------
    st.subheader("Quick Summary")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Base monthly EMI", fmt_usd(compute_emi(remaining_loan, annual_interest, months)))
        st.write("Loan amount")
        st.write(fmt_usd(remaining_loan))
    with s2:
        st.metric("First-year interest (base)", fmt_usd(first_year_interest_base))
        st.write("Annual property tax (est.)")
        st.write(fmt_usd(annual_property_tax))
    with s3:
        st.metric("Annual tax savings (base estimate)", fmt_usd(tax_savings_base))
        st.write("Deduction used")
        st.write("Itemized" if (first_year_interest_base + min(annual_property_tax, salt_cap)) > STANDARD_DEDUCTION_2025[filing_status] else "Standard")
    with s4:
        st.metric("Effective mortgage rate (base, 1st yr)", fmt_pct(effective_rate_base_pct))
        st.write("Effective (after lump)", f"{fmt_pct(effective_rate_lump_pct)} (after lump)")

    st.markdown("---")

    # ---------------------------
    # Investment vs Lump NPV comparison block
    # ---------------------------
    st.subheader("💰 Lump-sum Prepay vs Invest — Net Present Value (NPV) Comparison")
    st.caption(f"NPV is calculated by discounting future cash flows back to the present using the Expected Annual Return ({fmt_pct(annual_return)}). Higher NPV is better.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Option A: Prepay Mortgage**")
        st.write(f"Interest Saved by Prepaying: **{fmt_usd(interest_saved_by_lump)}**")
        st.write(f"Payoff accelerated by: **{months_saved_by_lump} months**")
        st.metric("NPV of Interest Savings (Prepay)", fmt_usd(npv_prepay), delta_color="off")


    with colB:
        st.markdown(f"**Option B: Invest for {invest_horizon_years} years**")
        inv_final_value = simulate_investment(lump_amount, monthly_invest=monthly_invest, annual_return_pct=annual_return, months=invest_months)
        st.write(f"Future Value after {invest_horizon_years} yrs: **{fmt_usd(inv_final_value)}**")
        st.metric("NPV of Investment Future Value (Invest)", fmt_usd(npv_invest), delta_color="off")


    # Recommendation
    st.markdown("---")
    st.subheader("🎯 Recommendation (Based on NPV)")
    if npv_invest > npv_prepay:
        st.success(f"**Investing is mathematically superior** by a Net Present Value of approx {fmt_usd(npv_invest - npv_prepay)}.")
        st.caption("This assumes the expected investment return is realized and ignores risk.")
    elif npv_prepay > npv_invest:
        st.info(f"**Prepaying the mortgage is mathematically superior** by a Net Present Value of approx {fmt_usd(npv_prepay - npv_invest)}.")
        st.caption("This suggests the guaranteed savings from the loan outweigh the potential returns from investing.")
    else:
        st.warning("The NPVs are approximately equal. Other factors (risk tolerance, liquidity needs) should decide.")

    st.markdown("---")

    # ---------------------------
    # Charts: Balances and Interest comparison
    # ---------------------------
    try:
        max_m = max(len(df_base), len(df_extra), len(df_lump))
        months_idx = np.arange(1, max_m + 1)
        df_plot = pd.DataFrame({"Month": months_idx})

        # Balance Data Preparation
        if not df_base.empty:
            df_plot = df_plot.merge(df_base[["Month","Balance"]].rename(columns={"Balance":"Balance_Base"}), on="Month", how="left")
        else:
            df_plot["Balance_Base"] = 0.0

        if not df_extra.empty:
            df_plot = df_plot.merge(df_extra[["Month","Balance"]].rename(columns={"Balance":"Balance_Extra"}), on="Month", how="left")
        else:
            df_plot["Balance_Extra"] = 0.0

        if not df_lump.empty:
            df_plot = df_plot.merge(df_lump[["Month","Balance"]].rename(columns={"Balance":"Balance_Lump"}), on="Month", how="left")
        else:
            df_plot["Balance_Lump"] = 0.0

        df_plot["Balance_Base"].ffill(inplace=True); df_plot["Balance_Base"].fillna(0,inplace=True)
        df_plot["Balance_Extra"].ffill(inplace=True); df_plot["Balance_Extra"].fillna(0,inplace=True)
        df_plot["Balance_Lump"].ffill(inplace=True); df_plot["Balance_Lump"].fillna(0,inplace=True)

        # 1. Balance Comparison Chart
        st.subheader("Outstanding Balance Over Time")
        melt = df_plot.melt("Month", value_vars=["Balance_Base","Balance_Extra","Balance_Lump"], var_name="Scenario", value_name="Balance")
        melt["Scenario"] = melt["Scenario"].map({"Balance_Base":"Base","Balance_Extra":"With Extra","Balance_Lump":"With Lump"})

        chart_balance = alt.Chart(melt).mark_line().encode(
            x="Month",
            y=alt.Y("Balance", title="Outstanding Balance ($)"),
            color="Scenario"
        ).properties(height=420)
        st.altair_chart(chart_balance, use_container_width=True)
        

        # 2. Cumulative Interest Chart
        st.subheader("Cumulative Interest Paid")

        # Interest Data Preparation
        df_plot["Interest_Base"] = df_base["Interest"].cumsum() if not df_base.empty else 0.0
        df_plot["Interest_Extra"] = df_extra["Interest"].cumsum() if not df_extra.empty else 0.0
        df_plot["Interest_Lump"] = df_lump["Interest"].cumsum() if not df_lump.empty else 0.0
        
        # Fill NA values for shorter amortization schedules after payoff
        df_plot["Interest_Base"].ffill(inplace=True); df_plot["Interest_Base"].fillna(0,inplace=True)
        df_plot["Interest_Extra"].ffill(inplace=True); df_plot["Interest_Extra"].fillna(0,inplace=True)
        df_plot["Interest_Lump"].ffill(inplace=True); df_plot["Interest_Lump"].fillna(0,inplace=True)
        
        melt_interest = df_plot.melt("Month", value_vars=["Interest_Base","Interest_Extra","Interest_Lump"], var_name="Scenario", value_name="Cumulative_Interest")
        melt_interest["Scenario"] = melt_interest["Scenario"].map({"Interest_Base":"Base","Interest_Extra":"With Extra","Interest_Lump":"With Lump"})

        chart_interest = alt.Chart(melt_interest).mark_line().encode(
            x="Month",
            y=alt.Y("Cumulative_Interest", title="Cumulative Interest Paid ($)"),
            color="Scenario"
        ).properties(height=420)
        st.altair_chart(chart_interest, use_container_width=True)
        

    except Exception:
        st.info("Ensure all dependencies (pandas, streamlit, numpy, altair, openpyxl) are installed to render charts.")

    st.markdown("---")

    # ---------------------------
    # Amortization tables & downloads (Rest of the file remains the same)
    # ---------------------------
    st.subheader("Amortization Schedules & Downloads")
    t1, t2, t3 = st.tabs(["Base schedule","With extra monthly","With lump"])

    with t1:
        if df_base.empty:
            st.write("No schedule")
        else:
            st.dataframe(df_base, use_container_width=True)
            st.download_button("Download base CSV", df_base.to_csv(index=False), "amortization_base.csv", "text/csv")
            st.download_button("Download base Excel", to_excel_bytes(df_base, "Base"), "amortization_base.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with t2:
        if df_extra.empty:
            st.write("No schedule")
        else:
            st.dataframe(df_extra, use_container_width=True)
            st.download_button("Download extra CSV", df_extra.to_csv(index=False), "amortization_extra.csv", "text/csv")
            st.download_button("Download extra Excel", to_excel_bytes(df_extra, "Extra"), "amortization_extra.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with t3:
        if df_lump.empty:
            st.write("No schedule")
        else:
            st.dataframe(df_lump, use_container_width=True)
            st.download_button("Download lump CSV", df_lump.to_csv(index=False), "amortization_lump.csv", "text/csv")
            st.download_button("Download lump Excel", to_excel_bytes(df_lump, "Lump"), "amortization_lump.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")

    # ---------------------------
    # Yearly summary (base)
    # ---------------------------
    def yearly_summary(df):
        if df.empty:
            return pd.DataFrame()
        df2 = df.copy()
        df2["Year"] = ((df2["Month"] - 1) // 12) + 1
        agg = df2.groupby("Year").agg({
            "Interest":"sum","PrincipalBase":"sum","ExtraPayment":"sum","TotalPayment":"sum","Balance":"last"
        }).reset_index()
        return agg

    st.subheader("Yearly summary (Base)")
    ys = yearly_summary(df_base)
    if ys.empty:
        st.write("No data")
    else:
        st.dataframe(ys, use_container_width=True)
        st.download_button("Download yearly CSV", ys.to_csv(index=False), "yearly_summary.csv", "text/csv")

    st.markdown("---")

    # ---------------------------
    # Tax details
    # ---------------------------
    st.subheader("Tax details (estimates) — 2025")
    st.write(f"Filing status: **{filing_status}**")
    st.write(f"Annual income: **{fmt_usd(income)}**")
    st.write(f"First-year mortgage interest (base): **{fmt_usd(first_year_interest_base)}**")
    st.write(f"First-year mortgage interest (with lump): **{fmt_usd(first_year_interest_lump)}**")
    st.write(f"Estimated annual tax savings (base): **{fmt_usd(tax_savings_base)}**")
    st.write(f"Estimated annual tax savings (with lump): **{fmt_usd(tax_savings_lump)}**")
    st.write(f"Estimated lost tax savings by prepaying: **{fmt_usd(lost_tax_savings_due_to_lump)}**")
    st.caption("Notes: Child tax credit modeled as $2,000 per dependent (simplified). SALT cap applied. For precise tax planning consult a tax professional.")