# Streamlit UI Guidelines

## Layout Rules
- All inputs go in the sidebar; all outputs go in the main area (full width)
- Use `st.columns` + `st.metric` for metric cards — never CSS class-based grid layouts
- Charts go side-by-side using `st.columns(2)`

## HTML / Markdown Rules
- Do NOT open a `<div>` in one `st.markdown` call and close it in another — each call renders in its own container
- Any custom HTML passed to `st.markdown(unsafe_allow_html=True)` must be a single self-contained block with inline styles only
- CSS classes only work reliably for simple single elements (e.g., hero banner); avoid for complex nested layouts

## Chart Rules
- Use `cumsum_padded(df, max_len)` when plotting cumulative series across schedules of different lengths
- Charts use Altair; do not switch to matplotlib or plotly

## Sections Order (main area)
1. Hero Banner
2. Quick Summary metrics
3. NPV Comparison cards + recommendation
4. Balance & Cumulative Interest charts
5. Amortization tables (tabs) with CSV/Excel downloads
6. Yearly Summary
7. Tax Details
