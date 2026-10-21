"""HomeValue — AI house price prediction, comparables and investment analysis (Streamlit + scikit-learn)."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import model as M
import theme
from ui import badge, hero, kpis, result_card, sidebar_brand

APP = dict(
    name="HomeValue", short="House price AI", icon="🏠", tagline="What is your home really worth?",
    description="Instant AI valuations for Bengaluru homes with price ranges, comparable sales, the factors behind every estimate and a 10-year investment outlook.",
    features=["Gradient Boosting, Random Forest, Ridge & MLP models (MAPE ≈ 8 %)", "80 % price range with quantile regression", "Comparable-sales finder & factor breakdown", "Locality heat map, rent-vs-buy and 10-year projections"],
    card1="🏠 3 BHK · HSR Layout · ₹1.92 Cr", card2="📈 Hebbal +8.2 % / year",
    demo_name="Bharath Kumar", demo_email="demo@homevalue.app", demo_password="Demo@1234",
    roles=["Home buyer", "Seller", "Agent", "Investor"],
)
st.set_page_config(page_title="HomeValue — House Price AI", page_icon="🏠", layout="wide")
theme.inject("Violet")


@st.cache_data(show_spinner="Loading 6,000 property transactions…")
def load():
    return M.generate()


@st.cache_resource(show_spinner="Training valuation models (~20 s, first run only)…")
def models():
    return M.train(load())


if "user" not in st.session_state:
    auth.auth_page(APP)
    st.stop()

user = st.session_state.user
sidebar_brand(APP, user)
PAGES = ["💰 Price estimator", "🗺️ Market map", "📈 Investment analyzer", "🤖 Model lab", "👤 Account"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
mname = st.sidebar.selectbox("🧠 Valuation model", list(M.MODELS))
theme.theme_controls()
if st.sidebar.button("Log out", width="stretch"):
    auth.logout()

D = load()
R = models()
LOC = pd.DataFrame([{"locality": k, "lat": v[0], "lon": v[1], "base_psf": v[2], "growth": v[3], "rent_psf": v[4], "tier": v[5]} for k, v in M.LOCALITIES.items()])
LOC = LOC.merge(D.groupby("locality").agg(median_psf=("price_psf", "median"), listings=("price", "size"), median_price=("price", "median")).reset_index())
LOC["yield %"] = LOC.rent_psf * 12 / LOC.median_psf * 100

# ------------------------------------------------------------------ Estimator
if page == PAGES[0]:
    hero("Instant home valuation", f"Fill in the details — {mname} estimates the price, range and why", "💰")
    c1, c2, c3 = st.columns(3)
    with c1:
        loc = st.selectbox("Locality", list(M.LOCALITIES), index=8)
        typ = st.selectbox("Property type", ["Apartment", "Villa", "Independent House"])
        bhk = st.select_slider("BHK", [1, 2, 3, 4, 5, 6], 3)
        sqft = st.slider("Built-up area (sq.ft)", 400, 6000, 1550, 10)
    with c2:
        bath = st.select_slider("Bathrooms", [1, 2, 3, 4, 5, 6], min(bhk, 3))
        age = st.slider("Age of property (years)", 0, 40, 4)
        tot = st.slider("Total floors in building", 1, 35, 14 if typ == "Apartment" else 2)
        floor = st.slider("Floor", 0, tot, min(5, tot))
    with c3:
        metro = st.slider("Distance to metro (km)", 0.2, 15.0, 1.8, 0.1)
        park = st.select_slider("Car parks", [0, 1, 2, 3], 1)
        amen = st.slider("Amenities score (0–10)", 0.0, 10.0, 7.5, 0.5, help="Gym, pool, clubhouse, security, power backup…")
        furn = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Fully furnished"], index=1)
        facing = st.selectbox("Facing", ["East", "North", "West", "South", "North-East"])
    row = dict(locality=loc, type=typ, bhk=bhk, sqft=sqft, bath=bath, age=age, floor=floor, total_floors=tot, metro_km=metro, parking=park, amenity_score=amen, furnishing=furn, facing=facing)
    p, lo, hi = M.predict(R, mname, row)
    lrow = LOC.set_index("locality").loc[loc]
    psf = p / sqft
    st.write("")
    a, b = st.columns([1.15, 1])
    with a:
        diff = (psf - lrow.median_psf) / lrow.median_psf * 100
        result_card("Estimated market value", M.fmt(p), f"80% range {M.fmt(lo)} – {M.fmt(hi)} · ₹{psf:,.0f}/sq.ft ({diff:+.0f}% vs {loc} median ₹{lrow.median_psf:,.0f})", "#7c3aed", "#1e1b4b", "🏠")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo / 1e5, hi / 1e5], y=[0, 0], mode="lines", line=dict(width=16, color="rgba(124,58,237,.25)"), showlegend=False))
        fig.add_trace(go.Scatter(x=[p / 1e5], y=[0], mode="markers+text", marker=dict(size=22, color="#7c3aed"), text=[M.fmt(p)], textposition="top center", showlegend=False))
        sub = D[(D.locality == loc) & (D.bhk == bhk)]
        if len(sub):
            fig.add_trace(go.Box(x=sub.price / 1e5, y=[-0.6] * len(sub), orientation="h", name=f"{bhk} BHK sales in {loc}", marker_color="#94a3b8", boxpoints=False))
        fig.update_layout(height=230, yaxis=dict(visible=False, range=[-1.2, 0.8]), xaxis_title="₹ Lakh", title="Your estimate vs. local sales", legend=dict(orientation="h", y=-0.4))
        st.plotly_chart(fig, width="stretch")
    with b:
        c = M.contributions(R, row).head(9)[::-1]
        fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=["#10b981" if v > 0 else "#ef4444" for v in c.values], text=[f"{v:+.1f}%" for v in c.values], textposition="auto"))
        fig.update_layout(title="What moves the price (vs. an average home)", height=400, xaxis_title="% effect")
        st.plotly_chart(fig, width="stretch")
    st.markdown("#### 🏘️ Comparable sales")
    comps = M.comparables(R, D, row)
    st.dataframe(comps.assign(price=comps.price.map(M.fmt), price_psf=comps.price_psf.map(lambda v: f"₹{v:,.0f}"))[["locality", "type", "bhk", "sqft", "bath", "age", "floor", "metro_km", "furnishing", "price", "price_psf"]], hide_index=True, width="stretch")
    k1, k2, k3, k4 = st.columns(4)
    loan = p * 0.8
    k1.metric("Monthly EMI (80% loan, 8.5%, 20y)", f"₹{M.emi(loan, 8.5, 20):,.0f}")
    k2.metric("Expected rent", f"₹{lrow.rent_psf * sqft:,.0f}/mo")
    k3.metric("Value in 5 years", M.fmt(p * (1 + lrow.growth) ** 5), f"+{lrow.growth * 100:.1f}%/yr")
    k4.metric("Stamp duty + reg. (~6.6%)", M.fmt(p * 0.066))
    report = "\n".join([f"HomeValue valuation report — {date.today()}", f"Prepared for {user['name']}", "", *[f"{k}: {v}" for k, v in row.items()], "", f"Estimated value: {M.fmt(p)}", f"80% range: {M.fmt(lo)} – {M.fmt(hi)}", f"Price per sq.ft: ₹{psf:,.0f}", f"Model: {mname}", "", "Comparable sales:", *[f"- {r.bhk} BHK, {r.sqft:.0f} sq.ft, {r.locality}: {M.fmt(r.price)}" for r in comps.itertuples()]])
    st.download_button("⬇️ Download valuation report", report, f"homevalue_report_{loc.replace(' ', '_')}.txt", type="primary")

# ------------------------------------------------------------------ Market map
elif page == PAGES[1]:
    hero("Bengaluru market map", f"{len(D):,} transactions across {len(M.LOCALITIES)} localities", "🗺️")
    metric = st.radio("Colour by", ["Median ₹/sq.ft", "Annual appreciation", "Rental yield"], horizontal=True)
    col = {"Median ₹/sq.ft": "median_psf", "Annual appreciation": "growth", "Rental yield": "yield %"}[metric]
    c1, c2 = st.columns([1.4, 1])
    with c1:
        samp = D.sample(1500, random_state=1)
        fig = px.scatter(samp, x="lon", y="lat", color="price_psf", color_continuous_scale="Turbo", opacity=0.35, size_max=6, hover_data=["locality", "bhk", "sqft"])
        fig.add_trace(go.Scatter(x=LOC.lon, y=LOC.lat, mode="markers+text", text=LOC.locality, textposition="top center", marker=dict(size=LOC[col] / LOC[col].max() * 40 + 10, color=LOC[col], colorscale="Viridis", line=dict(width=2, color="white")), showlegend=False,
                                 hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(height=560, title=f"Transactions (₹/sq.ft) with localities sized by {metric.lower()}", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), coloraxis_colorbar_title="₹/sq.ft")
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = LOC.sort_values(col, ascending=False)[["locality", "tier", "median_psf", "growth", "yield %", "listings"]]
        st.dataframe(t.assign(growth=(t.growth * 100).round(1), median_psf=t.median_psf.round(0), **{"yield %": t["yield %"].round(2)}).rename(columns={"median_psf": "₹/sq.ft", "growth": "growth %/yr"}), hide_index=True, width="stretch", height=560)
    c1, c2 = st.columns(2)
    fig = px.box(D, x="locality", y="price_psf", color="locality", title="₹/sq.ft distribution by locality")
    fig.update_layout(showlegend=False, xaxis_tickangle=-40, height=420)
    c1.plotly_chart(fig, width="stretch")
    fig = px.scatter(LOC, x="growth", y="yield %", size="median_psf", color="tier", text="locality", title="Appreciation vs rental yield")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=420, xaxis_tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Investment
elif page == PAGES[2]:
    hero("Investment analyzer", "Project value, rent and returns over 10 years — and compare buying with renting", "📈")
    c1, c2, c3 = st.columns(3)
    loc = c1.selectbox("Locality", list(M.LOCALITIES), index=6)
    price = c1.number_input("Purchase price (₹)", 2000000, 100000000, 12000000, 100000)
    down = c2.slider("Down payment %", 10, 100, 25)
    rate = c2.slider("Loan interest %", 6.5, 12.0, 8.5, 0.05)
    years = c3.slider("Loan tenure (years)", 5, 30, 20)
    lrow = LOC.set_index("locality").loc[loc]
    g = c3.slider("Expected appreciation %/yr", 0.0, 15.0, round(lrow.growth * 100, 1), 0.1) / 100
    sqft = price / lrow.median_psf
    rent0 = lrow.rent_psf * sqft
    loan = price * (1 - down / 100)
    e = M.emi(loan, rate, years) if loan > 0 else 0
    rows = []
    bal = loan
    for y in range(0, 11):
        val = price * (1 + g) ** y
        rows.append({"year": y, "value": val, "loan": max(bal, 0), "equity": val - max(bal, 0), "rent": rent0 * (1.05 ** y) * 12})
        for _ in range(12):
            bal = bal * (1 + rate / 1200) - e
    P = pd.DataFrame(rows)
    rent_total = P.rent.iloc[1:].sum()
    gain = P.value.iloc[-1] - price
    kpis([("🏦", "Monthly EMI", f"₹{e:,.0f}", f"loan {M.fmt(loan)}"), ("🏠", "Value in 10 yrs", M.fmt(P.value.iloc[-1]), f"+{M.fmt(gain)}"), ("💵", "Gross rental yield", f"{rent0 * 12 / price * 100:.2f}%", f"≈ ₹{rent0:,.0f}/mo"), ("📊", "Equity in 10 yrs", M.fmt(P.equity.iloc[-1]), "value − loan")])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P.year, y=P.value / 1e5, name="Property value", fill="tozeroy", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=P.year, y=P.loan / 1e5, name="Outstanding loan", line=dict(width=3, dash="dash")))
    fig.add_trace(go.Bar(x=P.year, y=P.rent / 1e5, name="Annual rent (if let out)", opacity=0.5))
    fig.update_layout(height=420, title="10-year projection (₹ Lakh)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🆚 Rent vs buy (10 years)")
        own_cost = e * 12 * min(10, years) + price * down / 100 + price * 0.066
        own_net = own_cost - P.equity.iloc[-1]
        rent_cost = rent_total
        inv = price * down / 100 * (1.11 ** 10) - price * down / 100
        fig = go.Figure(go.Bar(x=["Buy (net cost after equity)", "Rent (minus returns on invested down-payment)"], y=[own_net / 1e5, (rent_cost - inv) / 1e5], marker_color=["#7c3aed", "#f59e0b"], text=[M.fmt(own_net), M.fmt(rent_cost - inv)], textposition="outside"))
        fig.update_layout(height=340, yaxis_title="₹ Lakh")
        st.plotly_chart(fig, width="stretch")
        st.caption("Assumes rent grows 5 %/yr and the down payment would otherwise earn 11 %/yr in equity funds.")
    with c2:
        st.markdown("##### 🏆 Best localities for this budget")
        best = LOC.assign(sqft=price / LOC.median_psf, total_return=((1 + LOC.growth) ** 10 - 1) * 100 + LOC["yield %"] * 10).sort_values("total_return", ascending=False).head(6)
        fig = px.bar(best, x="total_return", y="locality", orientation="h", color="tier", text=best.sqft.map(lambda v: f"{v:,.0f} sq.ft"), title="10-yr appreciation + rent return (%)")
        fig.update_layout(height=340, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Model lab
elif page == PAGES[3]:
    hero("Model lab", "1,200 held-out sales · models trained on log(price)", "🤖")
    t = pd.DataFrame({n: {k: v[k] for k in ["MAE ₹L", "MAPE %", "R²"]} for n, v in R["models"].items()}).T
    st.dataframe(t.style.format("{:.3f}").highlight_min(subset=["MAE ₹L", "MAPE %"], color="#10b98144").highlight_max(subset=["R²"], color="#10b98144"), width="stretch")
    sel = R["models"][mname]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=R["yte"] / 1e5, y=sel["pred"] / 1e5, opacity=0.4, labels={"x": "actual (₹L)", "y": "predicted (₹L)"}, title=f"{mname}: predicted vs actual")
        mx = R["yte"].max() / 1e5
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dot"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    with c2:
        err = (sel["pred"] - R["yte"]) / R["yte"] * 100
        fig = px.histogram(err, nbins=60, title="Percentage error distribution")
        fig.add_vline(x=0, line_dash="dot")
        fig.update_layout(height=400, showlegend=False, xaxis_title="% error")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{(np.abs(err) <= 10).mean() * 100:.0f}% of estimates are within ±10 % of the actual price.")
    rf = R["models"]["Random Forest"]["pipe"]
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=[n.split("__", 1)[1] for n in rf.named_steps["pre"].get_feature_names_out()]).sort_values().tail(14)
    fig = px.bar(imp, orientation="h", title="Feature importance (Random Forest)")
    fig.update_layout(height=430, showlegend=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Account
else:
    hero(f"Hi, {user['name'].split()[0]} 👋", f"{user['email']} · {user['role']} · member since {user['created'][:10]}", "👤")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔒 Change password")
        with st.form("pw"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                try:
                    auth.change_password(user["email"], old, new)
                    st.success("Password updated")
                except ValueError as e:
                    st.error(str(e))
    with c2:
        st.markdown("##### 🗂️ Data")
        st.markdown(f'<div class="card">{len(D):,} simulated sales across {len(M.LOCALITIES)} Bengaluru localities with 13 features. Registered users: <b>{auth.user_count()}</b>.</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download dataset (CSV)", D.to_csv(index=False), "homevalue_sales.csv", "text/csv")
