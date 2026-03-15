# mortgage_tax_investment_app.py (Full code with CORRECTED NPV Investment Timing)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
import altair as alt

# ---------------------------
# Page config — MUST be first Streamlit call
# ---------------------------
st.set_page_config(
    page_title="Mortgage vs Invest Planner 2025",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
/* ── Overall background ── */
.stApp { background-color: #f0f4f8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3352 0%, #245785 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCheckbox span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div { color: #d4e8ff !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select { color: #1a3352 !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a3352 0%, #2471a3 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    color: white;
}
.hero h1 { color: white !important; font-size: 2rem; font-weight: 800; margin: 0 0 0.4rem 0; }
.hero p  { color: #a8cef0; margin: 0; font-size: 0.95rem; }

/* ── White content card ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}

/* ── Section heading inside cards ── */
.sec-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a3352;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e4edf7;
}

/* ── Metric badge cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.m-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 4px solid #2471a3;
}
.m-card.green  { border-top-color: #27ae60; }
.m-card.orange { border-top-color: #e67e22; }
.m-card.purple { border-top-color: #8e44ad; }
.m-label  { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.06em; color: #888; margin-bottom: 0.4rem; }
.m-value  { font-size: 1.55rem; font-weight: 800; color: #1a3352; line-height: 1; }
.m-sub    { font-size: 0.75rem; color: #999; margin-top: 0.35rem; }

/* ── NPV comparison cards ── */
.npv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1rem; }
.npv-card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center;
}
.npv-card.prepay { border-top: 5px solid #e67e22; }
.npv-card.invest  { border-top: 5px solid #27ae60; }
.npv-icon  { font-size: 2rem; margin-bottom: 0.5rem; }
.npv-label { font-size: 0.9rem; font-weight: 700; color: #555; margin-bottom: 0.8rem; }
.npv-val   { font-size: 2.2rem; font-weight: 900; margin-bottom: 0.5rem; }
.npv-card.prepay .npv-val { color: #e67e22; }
.npv-card.invest  .npv-val { color: #27ae60; }
.npv-detail { font-size: 0.82rem; color: #777; line-height: 1.5; }

/* ── Recommendation banner ── */
.rec { border-radius: 12px; padding: 1.1rem 1.5rem; margin: 0.8rem 0; font-size: 0.97rem; font-weight: 600; }
.rec.invest { background:#d5f5e3; color:#1a6e3c; border-left:5px solid #27ae60; }
.rec.prepay { background:#fef3e2; color:#7d4400; border-left:5px solid #e67e22; }
.rec.equal  { background:#eaf0fb; color:#1a3352; border-left:5px solid #2471a3; }
.rec-sub { font-size:0.82rem; font-weight:400; margin-top:0.35rem; color:inherit; opacity:0.8; }

/* ── Tax detail rows ── */
.tax-row { display:flex; justify-content:space-between; padding:0.5rem 0;
           border-bottom:1px solid #f0f4f8; font-size:0.88rem; }
.tax-row:last-child { border-bottom: none; }
.tax-key { color:#555; }
.tax-val { font-weight:600; color:#1a3352; }

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #2471a3 !important; color: white !important;
    border-radius: 8px !important; border: none !important;
    font-size: 0.78rem !important; padding: 0.35rem 0.9rem !important;
}
.stDownloadButton > button:hover { background: #1a5a8a !important; }

/* ── Sidebar sub-heading ── */
.sb-hdr { font-size:0.68rem; font-weight:800; text-transform:uppercase;
          letter-spacing:0.1em; color:#7aafd4 !important; margin-top:0.8rem; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: #f0f4f8; border-radius: 10px; padding: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; font-weight: 600 !important; }
[aria-selected="true"][data-baseweb="tab"] { background: white !important; color: #2471a3 !important; }

/* ── Hide default footer / menu ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Formatting helpers
# ---------------------------
def fmt_usd(x):
    try:    return f"${float(x):,.2f}"
    except: return str(x)

def fmt_pct(x):
    try:    return f"{x:.2f}%"
    except: return str(x)

def to_excel_bytes(df, sheet_name="Sheet1"):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return out.getvalue()

def npv_card_html(icon, title, npv_val, detail_html, accent):
    return f"""
    <div style="background:white;border-radius:14px;padding:1.6rem;
                box-shadow:0 2px 10px rgba(0,0,0,0.08);text-align:center;
                border-top:5px solid {accent}; height:100%;">
      <div style="font-size:2rem;margin-bottom:0.4rem;">{icon}</div>
      <div style="font-size:0.88rem;font-weight:700;color:#555;margin-bottom:0.8rem;">{title}</div>
      <div style="font-size:2rem;font-weight:900;color:{accent};margin-bottom:0.5rem;">{npv_val}</div>
      <div style="font-size:0.82rem;color:#777;line-height:1.6;">{detail_html}</div>
    </div>"""

# ---------------------------
# 2025 standard deductions
# ---------------------------
STANDARD_DEDUCTION_2025 = {
    "Single": 15750.0,
    "Married Filing Separately": 15750.0,
    "Head of Household": 23625.0,
    "Married Filing Jointly": 31500.0,
    "Surviving Spouses": 31500.0,
}

# ---------------------------
# 2025 federal tax brackets
# ---------------------------
FEDERAL_BRACKETS_2025 = {
    "Single": [
        (0, 11925, 0.10), (11925, 48475, 0.12), (48475, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250525, 0.32),
        (250525, 626350, 0.35), (626350, float("inf"), 0.37),
    ],
    "Married Filing Jointly": [
        (0, 23850, 0.10), (23850, 96950, 0.12), (96950, 206700, 0.22),
        (206700, 394600, 0.24), (394600, 501050, 0.32),
        (501050, 751600, 0.35), (751600, float("inf"), 0.37),
    ],
    "Married Filing Separately": [
        (0, 11925, 0.10), (11925, 48475, 0.12), (48475, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250525, 0.32),
        (250525, 626350, 0.35), (626350, float("inf"), 0.37),
    ],
    "Head of Household": [
        (0, 17000, 0.10), (17000, 64850, 0.12), (64850, 103350, 0.22),
        (103350, 197300, 0.24), (197300, 250500, 0.32),
        (250500, 626350, 0.35), (626350, float("inf"), 0.37),
    ],
    "Surviving Spouses": [
        (0, 23850, 0.10), (23850, 96950, 0.12), (96950, 206700, 0.22),
        (206700, 394600, 0.24), (394600, 501050, 0.32),
        (501050, 751600, 0.35), (751600, float("inf"), 0.37),
    ],
}

# ---------------------------
# Finance core functions (unchanged)
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

def compute_emi(P, annual_rate_pct, n_months):
    r = annual_rate_pct / 100.0 / 12.0
    if n_months <= 0: return 0.0
    if r == 0:        return P / n_months
    powv = (1 + r) ** n_months
    return P * r * powv / (powv - 1)

def amortization_schedule(P, annual_rate_pct, n_months, extra_monthly=0.0):
    monthly_r = annual_rate_pct / 100.0 / 12.0
    base_emi  = compute_emi(P, annual_rate_pct, n_months)
    balance   = float(P)
    month, rows = 0, []
    cap = max(n_months * 10, 1200)
    while balance > 0.005 and month < cap:
        month += 1
        interest       = balance * monthly_r
        principal_base = max(0.0, base_emi - interest)
        extra          = min(extra_monthly, max(0.0, balance - principal_base))
        payment        = base_emi + extra
        if principal_base + extra >= balance - 1e-9:
            principal_paid = balance
            payment        = interest + principal_paid
            balance        = 0.0
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(payment,2), round(balance,2)])
            break
        else:
            principal_paid = principal_base + extra
            balance       -= principal_paid
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(payment,2), round(balance,2)])
    return pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])

def apply_lump_and_resimulate(original_schedule_df, P, annual_rate_pct, n_months, lump_amount, lump_month, extra_monthly=0.0):
    monthly_r = annual_rate_pct / 100.0 / 12.0
    base_emi  = compute_emi(P, annual_rate_pct, n_months)
    balance   = float(P)
    month, rows = 0, []
    cap = max(n_months * 10, 1200)
    while balance > 0.005 and month < cap:
        month  += 1
        interest       = balance * monthly_r
        principal_base = max(0.0, base_emi - interest)
        extra          = 0.0
        if principal_base >= balance:
            rows.append([month, round(base_emi,2), round(interest,2), round(balance,2), 0.0, round(interest+balance,2), 0.0])
            balance = 0.0
            break
        else:
            balance -= principal_base
            rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), 0.0, round(base_emi,2), round(balance,2)])
        if month == lump_month:
            applied = min(lump_amount, balance)
            balance -= applied
            if balance <= 0.005:
                month += 1
                rows.append([month, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                break
    if balance > 0.005 and month < cap:
        while balance > 0.005 and month < cap:
            month  += 1
            interest       = balance * monthly_r
            principal_base = max(0.0, base_emi - interest)
            extra          = min(extra_monthly, max(0.0, balance - principal_base)) if extra_monthly > 0 else 0.0
            if principal_base + extra >= balance - 1e-9:
                principal_paid = balance
                balance        = 0.0
                rows.append([month, round(base_emi,2), round(interest,2), round(principal_paid,2), round(extra,2), round(interest+principal_paid,2), 0.0])
                break
            else:
                principal_paid  = principal_base + extra
                balance        -= principal_paid
                rows.append([month, round(base_emi,2), round(interest,2), round(principal_base,2), round(extra,2), round(base_emi+extra,2), round(balance,2)])
    return pd.DataFrame(rows, columns=["Month","BaseEMI","Interest","PrincipalBase","ExtraPayment","TotalPayment","Balance"])

def simulate_investment(lump_amount, monthly_invest=0.0, annual_return_pct=10.0, months=12):
    r_month = (1 + annual_return_pct/100.0) ** (1/12.0) - 1.0
    balance = float(lump_amount)
    for _ in range(1, months + 1):
        balance = balance * (1 + r_month)
        if monthly_invest > 0:
            balance += monthly_invest
    return balance

def compute_annual_tax_savings(first_year_interest, income, filing_status, num_dependents,
                               include_state, state_tax_rate, annual_property_tax, salt_cap=10000.0):
    std      = STANDARD_DEDUCTION_2025.get(filing_status, 0.0)
    salt     = min(annual_property_tax, salt_cap)
    itemized = first_year_interest + salt
    taxable_standard = max(0.0, income - std)
    taxable_itemized = max(0.0, income - itemized)
    tax_standard = compute_progressive_tax(taxable_standard, filing_status)
    tax_itemized = compute_progressive_tax(taxable_itemized, filing_status)
    credit = num_dependents * 2000.0
    federal_savings = max(0.0, tax_standard - credit) - max(0.0, tax_itemized - credit)
    state_savings   = max(0.0, itemized - std) * state_tax_rate if include_state else 0.0
    total           = max(0.0, federal_savings + state_savings)
    return total, federal_savings, state_savings

def calculate_npv(df_base, df_scenario, lump_amount, monthly_invest, annual_r_invest,
                  invest_horizon_months, interest_saved_total, tax_savings_annual_base,
                  tax_savings_annual_scenario, lump_month):
    r_month     = (1 + annual_r_invest / 100.0) ** (1/12.0) - 1.0
    max_m       = max(len(df_base), len(df_scenario))
    base_idx    = df_base.set_index("Month").reindex(range(1, max_m+1), fill_value=0.0).reset_index(names=["Month"])
    scen_idx    = df_scenario.set_index("Month").reindex(range(1, max_m+1), fill_value=0.0).reset_index(names=["Month"])
    saved_monthly = base_idx["Interest"] - scen_idx["Interest"]
    npv_prepay  = np.sum([saved_monthly.iloc[t] / ((1+r_month)**(t+1)) for t in range(len(saved_monthly))])
    delay_months       = max(0, lump_month - 1)
    compounding_months = max(0, invest_horizon_months - delay_months)
    if compounding_months <= 0:
        inv_fv = lump_amount
    else:
        inv_fv = simulate_investment(lump_amount, monthly_invest=monthly_invest,
                                     annual_return_pct=annual_r_invest, months=compounding_months)
    total_time = delay_months + compounding_months
    npv_invest  = inv_fv / ((1+r_month)**total_time) if total_time > 0 else inv_fv
    return npv_prepay, npv_invest

# ============================================================
# SIDEBAR — All inputs
# ============================================================
with st.sidebar:
    st.markdown("## 🏦 Planner Inputs")

    st.markdown('<p class="sb-hdr">Mortgage</p>', unsafe_allow_html=True)
    home_price      = st.number_input("Home price ($)",            value=500000.0, step=1000.0,  format="%.2f")
    down_payment    = st.number_input("Down payment ($)",          value=100000.0, step=1000.0,  format="%.2f")
    remaining_loan  = max(0.0, home_price - down_payment)
    st.caption(f"Loan amount: **{fmt_usd(remaining_loan)}**")
    annual_interest = st.number_input("Annual interest rate (%)",  value=6.5,      step=0.01,    format="%.2f")
    remaining_years = st.number_input("Tenure (years)",            value=20,       min_value=0,  step=1)
    extra_monthly   = st.number_input("Extra monthly payment ($)", value=0.0,      step=50.0,    format="%.2f")
    property_tax_rt = st.number_input("Property tax rate (%)",     value=1.0,      step=0.01,    format="%.2f") / 100.0

    st.markdown('<p class="sb-hdr">Tax (US 2025)</p>', unsafe_allow_html=True)
    filing_status   = st.selectbox("Filing status", list(STANDARD_DEDUCTION_2025.keys()), index=3)
    income          = st.number_input("Annual gross income ($)",   value=150000.0, step=1000.0,  format="%.2f")
    num_dependents  = st.number_input("Dependents (children)",     value=0,        min_value=0,  step=1)
    include_state   = st.checkbox("Include state tax (SALT approx)", value=False)
    state_tax_rate  = st.number_input("State tax rate (%)",        value=5.0,      step=0.1,     format="%.2f") / 100.0 if include_state else 0.0
    salt_cap        = st.number_input("SALT cap ($)",              value=10000.0,  step=100.0,   format="%.2f")

    st.markdown('<p class="sb-hdr">Lump-sum & Investment</p>', unsafe_allow_html=True)
    lump_amount       = st.number_input("Lump-sum amount ($)",         value=20000.0, step=100.0, format="%.2f")
    lump_month        = st.number_input("Lump-sum month (1 = now)",    value=12,      min_value=1, step=1)
    monthly_invest    = st.number_input("Monthly invest amount ($)",   value=0.0,     step=10.0,  format="%.2f")
    annual_return     = st.number_input("Expected annual return (%)",  value=10.0,    step=0.1,   format="%.2f")
    invest_horizon_yr = st.number_input("Investment horizon (years)",  value=remaining_years, min_value=1, step=1)

    st.markdown("---")
    st.caption("💡 For precise tax planning, consult a tax professional. This tool provides estimates only.")

# ============================================================
# COMPUTE all schedules & metrics
# ============================================================
months       = int(max(1, round(remaining_years * 12)))
df_base      = amortization_schedule(remaining_loan, annual_interest, months)
df_extra     = amortization_schedule(remaining_loan, annual_interest, months, extra_monthly=extra_monthly or 0.0)
df_lump      = apply_lump_and_resimulate(df_base, remaining_loan, annual_interest, months,
                                         lump_amount, lump_month, extra_monthly=extra_monthly or 0.0)

total_interest_base   = df_base["Interest"].sum() if not df_base.empty else 0.0
total_interest_lump   = df_lump["Interest"].sum() if not df_lump.empty else 0.0
first_yr_int_base     = df_base.loc[df_base["Month"] <= 12, "Interest"].sum() if not df_base.empty else 0.0
first_yr_int_lump     = df_lump.loc[df_lump["Month"] <= 12, "Interest"].sum() if not df_lump.empty else 0.0
months_base, months_lump = len(df_base), len(df_lump)
months_saved          = max(0, months_base - months_lump)
interest_saved        = max(0.0, total_interest_base - total_interest_lump)

annual_property_tax   = home_price * property_tax_rt
tax_sav_base,  fed_s_base, st_s_base = compute_annual_tax_savings(
    first_yr_int_base, income, filing_status, num_dependents, include_state, state_tax_rate, annual_property_tax, salt_cap)
tax_sav_lump,  fed_s_lump, st_s_lump = compute_annual_tax_savings(
    first_yr_int_lump, income, filing_status, num_dependents, include_state, state_tax_rate, annual_property_tax, salt_cap)
lost_tax_sav = max(0.0, tax_sav_base - tax_sav_lump)

invest_months = int(invest_horizon_yr * 12)
npv_prepay, npv_invest = calculate_npv(
    df_base, df_lump, lump_amount, monthly_invest, annual_return,
    invest_months, interest_saved, tax_sav_base, tax_sav_lump, lump_month)

emi_base = compute_emi(remaining_loan, annual_interest, months)
eff_int_base  = first_yr_int_base - tax_sav_base
eff_rate_base = (eff_int_base / remaining_loan * 100.0) if remaining_loan > 0 else 0.0

delay_months       = max(0, lump_month - 1)
compounding_months = max(0, invest_months - delay_months)
inv_fv = lump_amount if compounding_months <= 0 else simulate_investment(
    lump_amount, monthly_invest=monthly_invest, annual_return_pct=annual_return, months=compounding_months)

uses_itemized = (first_yr_int_base + min(annual_property_tax, salt_cap)) > STANDARD_DEDUCTION_2025[filing_status]

# ============================================================
# MAIN OUTPUT
# ============================================================

# ── Hero ──────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🏦 Mortgage vs. Investment Planner</h1>
  <p>2025 US Federal Tax &nbsp;·&nbsp; Amortization Analysis &nbsp;·&nbsp; NPV Comparison
  &nbsp;|&nbsp; Loan: <strong>{fmt_usd(remaining_loan)}</strong>
  @ <strong>{annual_interest}%</strong> for <strong>{remaining_years} yrs</strong></p>
</div>
""", unsafe_allow_html=True)

# ── Quick-summary metrics (native st.metric + CSS) ────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Monthly EMI",             fmt_usd(emi_base),          f"Loan: {fmt_usd(remaining_loan)}")
m2.metric("1st-Year Interest",       fmt_usd(first_yr_int_base), f"Prop. tax: {fmt_usd(annual_property_tax)}")
m3.metric("Annual Tax Savings",      fmt_usd(tax_sav_base),      f"{'Itemized' if uses_itemized else 'Standard'} deduction")
m4.metric("Effective Rate (yr 1)",   fmt_pct(eff_rate_base),     "After tax savings")

st.markdown("---")

# ── NPV Comparison ────────────────────────────────────────
st.subheader("💰 Lump-sum Prepay vs. Invest — NPV Comparison")
st.caption(f"Present values discounted at the expected investment return ({fmt_pct(annual_return)}). Higher NPV = better outcome.")

npv_c1, npv_c2 = st.columns(2)
with npv_c1:
    st.markdown(npv_card_html(
        "🏠", "Option A — Prepay Mortgage",
        fmt_usd(npv_prepay),
        f"Interest saved: <strong>{fmt_usd(interest_saved)}</strong><br>Payoff accelerated: <strong>{months_saved} months</strong>",
        "#e67e22"
    ), unsafe_allow_html=True)

with npv_c2:
    st.markdown(npv_card_html(
        "📈", f"Option B — Invest for {invest_horizon_yr} yrs",
        fmt_usd(npv_invest),
        f"Future value: <strong>{fmt_usd(inv_fv)}</strong><br>Compounding: <strong>{compounding_months} months</strong>",
        "#27ae60"
    ), unsafe_allow_html=True)

# Recommendation
st.markdown("")
if npv_invest > npv_prepay:
    diff = fmt_usd(npv_invest - npv_prepay)
    st.success(f"**Investing is mathematically superior** by ~{diff} NPV.  \nAssumes the expected return is realised. Investing carries market risk — guaranteed mortgage savings are risk-free.")
elif npv_prepay > npv_invest:
    diff = fmt_usd(npv_prepay - npv_invest)
    st.info(f"**Prepaying the mortgage is mathematically superior** by ~{diff} NPV.  \nThe guaranteed interest savings outweigh projected investment returns at this assumed rate.")
else:
    st.warning("**The NPVs are approximately equal.** Risk tolerance and liquidity needs should guide your decision.")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────
try:
    max_m      = max(len(df_base), len(df_extra), len(df_lump))
    months_idx = np.arange(1, max_m + 1)
    df_plot    = pd.DataFrame({"Month": months_idx})

    for df_src, col in [(df_base, "Balance_Base"), (df_extra, "Balance_Extra"), (df_lump, "Balance_Lump")]:
        if not df_src.empty:
            df_plot = df_plot.merge(df_src[["Month","Balance"]].rename(columns={"Balance": col}), on="Month", how="left")
        else:
            df_plot[col] = 0.0

    for col in ["Balance_Base","Balance_Extra","Balance_Lump"]:
        df_plot[col] = df_plot[col].ffill().fillna(0)

    def cumsum_padded(df, max_len):
        if df.empty:
            return np.zeros(max_len)
        arr = df["Interest"].cumsum().values
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), mode="edge")
        return arr[:max_len]

    df_plot["Interest_Base"]  = cumsum_padded(df_base,  max_m)
    df_plot["Interest_Extra"] = cumsum_padded(df_extra, max_m)
    df_plot["Interest_Lump"]  = cumsum_padded(df_lump,  max_m)

    CHART_COLORS = {"Base":"#2471a3","With Extra":"#27ae60","With Lump":"#e67e22"}

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Outstanding Balance")
        melt_bal = df_plot.melt("Month",
            value_vars=["Balance_Base","Balance_Extra","Balance_Lump"],
            var_name="Scenario", value_name="Balance")
        melt_bal["Scenario"] = melt_bal["Scenario"].map(
            {"Balance_Base":"Base","Balance_Extra":"With Extra","Balance_Lump":"With Lump"})
        st.altair_chart(
            alt.Chart(melt_bal).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Month:Q"),
                y=alt.Y("Balance:Q", title="Balance ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Scenario:N", scale=alt.Scale(
                    domain=list(CHART_COLORS.keys()), range=list(CHART_COLORS.values()))),
                tooltip=["Month","Scenario", alt.Tooltip("Balance:Q", format="$,.2f")]
            ).properties(height=350),
            use_container_width=True)

    with c2:
        st.subheader("📈 Cumulative Interest")
        melt_int = df_plot.melt("Month",
            value_vars=["Interest_Base","Interest_Extra","Interest_Lump"],
            var_name="Scenario", value_name="Cumulative_Interest")
        melt_int["Scenario"] = melt_int["Scenario"].map(
            {"Interest_Base":"Base","Interest_Extra":"With Extra","Interest_Lump":"With Lump"})
        st.altair_chart(
            alt.Chart(melt_int).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Month:Q"),
                y=alt.Y("Cumulative_Interest:Q", title="Cumul. Interest ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Scenario:N", scale=alt.Scale(
                    domain=list(CHART_COLORS.keys()), range=list(CHART_COLORS.values()))),
                tooltip=["Month","Scenario", alt.Tooltip("Cumulative_Interest:Q", format="$,.2f")]
            ).properties(height=350),
            use_container_width=True)

except Exception as e:
    st.warning(f"Chart rendering error: {e}")

st.markdown("---")

# ── Amortization Tables ───────────────────────────────────
st.subheader("📋 Amortization Schedules & Downloads")
t1, t2, t3 = st.tabs(["  Base Schedule  ", "  With Extra Monthly  ", "  With Lump Sum  "])

for tab, df_tab, label in [(t1, df_base, "base"), (t2, df_extra, "extra"), (t3, df_lump, "lump")]:
    with tab:
        if df_tab.empty:
            st.info("No schedule generated.")
        else:
            st.dataframe(df_tab.style.format({
                "BaseEMI":"${:,.2f}", "Interest":"${:,.2f}", "PrincipalBase":"${:,.2f}",
                "ExtraPayment":"${:,.2f}", "TotalPayment":"${:,.2f}", "Balance":"${:,.2f}"
            }), use_container_width=True, height=320)
            bc1, bc2 = st.columns(2)
            with bc1:
                st.download_button("⬇ Download CSV", df_tab.to_csv(index=False),
                                   f"amortization_{label}.csv", "text/csv")
            with bc2:
                st.download_button("⬇ Download Excel", to_excel_bytes(df_tab, label.title()),
                                   f"amortization_{label}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ── Yearly Summary + Tax Details (side by side) ───────────
col_y, col_t = st.columns([3, 2])

with col_y:
    st.subheader("📅 Yearly Summary (Base)")
    if not df_base.empty:
        df2 = df_base.copy()
        df2["Year"] = ((df2["Month"] - 1) // 12) + 1
        ys = df2.groupby("Year").agg(
            Interest=("Interest","sum"), Principal=("PrincipalBase","sum"),
            ExtraPayment=("ExtraPayment","sum"), TotalPayment=("TotalPayment","sum"),
            Balance=("Balance","last")
        ).reset_index()
        st.dataframe(ys.style.format({
            "Interest":"${:,.2f}", "Principal":"${:,.2f}", "ExtraPayment":"${:,.2f}",
            "TotalPayment":"${:,.2f}", "Balance":"${:,.2f}"
        }), use_container_width=True, height=300)
        st.download_button("⬇ Yearly CSV", ys.to_csv(index=False), "yearly_summary.csv", "text/csv")

with col_t:
    st.subheader("🧾 Tax Details (2025 Estimates)")
    tax_rows = [
        ("Filing status",               filing_status),
        ("Annual income",               fmt_usd(income)),
        ("1st-yr interest (base)",      fmt_usd(first_yr_int_base)),
        ("1st-yr interest (with lump)", fmt_usd(first_yr_int_lump)),
        ("Federal savings (base)",      fmt_usd(fed_s_base)),
        ("State savings (base)",        fmt_usd(st_s_base) if include_state else "N/A"),
        ("Total tax savings (base)",    fmt_usd(tax_sav_base)),
        ("Total tax savings (lump)",    fmt_usd(tax_sav_lump)),
        ("Lost savings from prepay",    fmt_usd(lost_tax_sav)),
        ("Deduction type",              "Itemized" if uses_itemized else "Standard"),
    ]
    for k, v in tax_rows:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:0.45rem 0;'
            f'border-bottom:1px solid #eef2f7;font-size:0.88rem;">'
            f'<span style="color:#666;">{k}</span>'
            f'<span style="font-weight:600;color:#1a3352;">{v}</span></div>',
            unsafe_allow_html=True)
    st.caption("Child tax credit: $2,000/dependent (simplified). SALT cap applied. Consult a CPA for precise planning.")
