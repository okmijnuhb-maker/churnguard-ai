import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | Telecom Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, body, .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    box-sizing: border-box;
}
.stApp {
    background-color: #0d1117;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #d1d5db !important; }
[data-testid="stSidebar"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown b,
[data-testid="stSidebar"] .stMarkdown strong {
    color: #f9fafb !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] hr { border-color: #1f2937 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stTextInput > label {
    color: #9ca3af !important;
    font-size: 12px !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: #1f2937 !important;
    color: #f9fafb !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00bfa5 !important;
    box-shadow: 0 0 0 2px rgba(0,191,165,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280 !important;
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
    padding: 7px 18px;
    letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
    background: #00bfa5 !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #00bfa5;
    color: #0d1117 !important;
    border: none;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    padding: 10px 20px;
    width: 100%;
    letter-spacing: 0.03em;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #00d4b8 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f9fafb !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 20px 22px;
    margin: 8px 0;
}
.card-accent-teal  { border-left: 3px solid #00bfa5; }
.card-accent-red   { border-left: 3px solid #ef4444; }
.card-accent-amber { border-left: 3px solid #f59e0b; }
.card-accent-green { border-left: 3px solid #22c55e; }

/* ── Risk banners ── */
.risk-high {
    background: linear-gradient(135deg, #1c0a0a, #2d0f0f);
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 28px 32px;
}
.risk-medium {
    background: linear-gradient(135deg, #1c1408, #2d1f06);
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 28px 32px;
}
.risk-low {
    background: linear-gradient(135deg, #071c10, #0a2917);
    border: 1px solid #14532d;
    border-radius: 10px;
    padding: 28px 32px;
}

/* ── Section header ── */
.section-header {
    background: #111827;
    border-left: 3px solid #00bfa5;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    margin: 18px 0 12px 0;
}
.section-header h4 {
    margin: 0;
    color: #f9fafb !important;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Typography ── */
h1, h2, h3, h4 { color: #f9fafb !important; }
p, li { color: #d1d5db !important; font-size: 14px; }
hr { border-color: #1f2937 !important; }

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #1f2937 !important;
    color: #d1d5db !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    transition: border-color 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #00bfa5 !important;
    color: #00bfa5 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #374151;
    border-radius: 8px;
    padding: 8px;
}

/* ── Recommendation items ── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #1f2937;
}
.rec-item:last-child { border-bottom: none; }
.rec-priority {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 7px;
    border-radius: 4px;
    white-space: nowrap;
    margin-top: 1px;
}
.rec-priority-high   { background: #7f1d1d; color: #fca5a5; }
.rec-priority-medium { background: #78350f; color: #fcd34d; }
.rec-priority-watch  { background: #1e3a5f; color: #93c5fd; }
.rec-text {
    color: #d1d5db !important;
    font-size: 13px;
    line-height: 1.5;
    margin: 0;
}
.rec-text b { color: #f9fafb !important; }

/* ── Profile table ── */
.profile-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1a2232;
    font-size: 13px;
}
.profile-row:last-child { border-bottom: none; }
.profile-key   { color: #6b7280; }
.profile-value { color: #f9fafb; font-weight: 500; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    mdl = joblib.load("models/churn_model.pkl")
    scl = joblib.load("models/scaler.pkl")
    with open("models/feature_cols.json") as f:
        cols = json.load(f)
    return mdl, scl, cols

model, scaler, feature_cols = load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_input(tenure, monthly_charges, total_charges, senior_citizen,
                partner, dependents, gender, internet, phone, multiple_lines,
                online_security, online_backup, device_protect, tech_support,
                streaming_tv, streaming_movies, contract, paperless, payment):
    raw = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges, 'gender': gender,
        'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone, 'MultipleLines': multiple_lines,
        'InternetService': internet, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protect,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
    }
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_cols, fill_value=0)
    df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
        df[['tenure','MonthlyCharges','TotalCharges']])
    return df


def get_risk_meta(prob):
    """Return (label, color, css_class) based on probability."""
    if prob >= 0.7:
        return "HIGH RISK", "#ef4444", "risk-high"
    elif prob >= 0.4:
        return "MEDIUM RISK", "#f59e0b", "risk-medium"
    else:
        return "LOW RISK", "#22c55e", "risk-low"


def build_recommendations(prob, tenure, contract, internet, payment,
                           monthly_charges, senior_citizen, tech_support,
                           online_security, multiple_lines):
    """
    Returns a list of dicts: {priority, title, detail}
    Priority: 'HIGH' | 'MEDIUM' | 'WATCH'
    Recommendations are derived from the actual customer profile — not generic.
    """
    recs = []

    # ── Contract ──────────────────────────────────────────────────────────────
    if contract == "Month-to-month":
        disc = "15–20%" if prob >= 0.7 else "10–15%"
        recs.append({
            "priority": "HIGH" if prob >= 0.5 else "MEDIUM",
            "title": "Offer contract upgrade",
            "detail": f"Customer is on a month-to-month plan — the highest-churn contract type (42.7% churn rate). "
                      f"Offer a one- or two-year contract with a {disc} discount on monthly charges to lock in commitment."
        })
    elif contract == "One year" and prob >= 0.4:
        recs.append({
            "priority": "MEDIUM",
            "title": "Upgrade to two-year contract",
            "detail": "Customer is on a one-year plan. Proactively offer a two-year renewal at renewal time with "
                      "a loyalty rate — two-year contracts show only 2.8% churn vs 42.7% for monthly."
        })

    # ── Tenure ────────────────────────────────────────────────────────────────
    if tenure < 12:
        recs.append({
            "priority": "HIGH" if prob >= 0.5 else "MEDIUM",
            "title": "Early lifecycle intervention",
            "detail": f"Customer has only {tenure} month(s) of tenure — early churn window (0–12 months shows 47.4% churn). "
                      f"Assign a dedicated onboarding contact, confirm service setup is complete, and schedule a 30-day check-in call."
        })
    elif tenure < 24 and prob >= 0.5:
        recs.append({
            "priority": "MEDIUM",
            "title": "Loyalty recognition at 24-month milestone",
            "detail": f"Customer is at {tenure} months. Approaching the 24-month mark is a natural retention moment — "
                      f"send a loyalty acknowledgement and offer a service bundle upgrade or bill credit."
        })

    # ── Internet / Fiber ──────────────────────────────────────────────────────
    if internet == "Fiber optic" and prob >= 0.4:
        recs.append({
            "priority": "HIGH" if prob >= 0.7 else "MEDIUM",
            "title": "Fiber optic service quality review",
            "detail": "Fiber optic customers churn at 41% — the highest of any internet segment. "
                      "Proactively audit the customer's service quality and ticket history. If there are unresolved issues, escalate. "
                      "If the bill is high, offer a speed downgrade with cost savings or a promotional rate."
        })

    # ── Payment method ────────────────────────────────────────────────────────
    if payment == "Electronic check":
        recs.append({
            "priority": "MEDIUM",
            "title": "Migrate to automatic payment",
            "detail": "Electronic check is the highest-churn payment method. Offer a $5–10/month bill credit "
                      "or one-time incentive to switch to bank transfer or credit card auto-pay. "
                      "This reduces payment friction and is correlated with lower churn rates."
        })

    # ── Tech Support ──────────────────────────────────────────────────────────
    if tech_support == "No" and internet != "No" and prob >= 0.4:
        recs.append({
            "priority": "MEDIUM",
            "title": "Offer tech support trial",
            "detail": "Customer currently has no tech support. Customers without tech support churn at significantly higher rates. "
                      "Offer a 3-month complimentary trial — once the service is in use, adoption tends to be sticky."
        })

    # ── Online Security ───────────────────────────────────────────────────────
    if online_security == "No" and internet != "No" and prob >= 0.5:
        recs.append({
            "priority": "WATCH",
            "title": "Bundle online security service",
            "detail": "No online security on record. Bundle online security with a modest price increase and frame it as a "
                      "proactive upgrade — adds perceived value and increases the number of active services, which lowers churn propensity."
        })

    # ── Monthly charges ───────────────────────────────────────────────────────
    if monthly_charges > 90 and prob >= 0.5:
        recs.append({
            "priority": "MEDIUM" if prob >= 0.6 else "WATCH",
            "title": "Review billing relative to services used",
            "detail": f"Monthly charges of ${monthly_charges:.0f} are in the high-cost tier. "
                      f"Review whether the services on the account justify the cost. If the customer is paying for unused services, "
                      f"a right-sizing conversation can prevent churn better than a generic retention call."
        })

    # ── Senior Citizen ────────────────────────────────────────────────────────
    if senior_citizen == "Yes" and prob >= 0.4:
        recs.append({
            "priority": "WATCH",
            "title": "Senior-specific outreach",
            "detail": "Senior citizens show higher churn rates in this dataset. Consider a dedicated support call "
                      "to verify billing clarity, ease of use of services, and whether a simplified plan better fits the customer's needs."
        })

    # ── Low risk upsell ───────────────────────────────────────────────────────
    if prob < 0.4:
        if multiple_lines == "No" and internet != "No":
            recs.append({
                "priority": "WATCH",
                "title": "Upsell opportunity — multiple lines",
                "detail": "Customer is stable and low-risk. This is a good time to introduce a multiple lines bundle, "
                          "particularly if the customer has a partner or dependents. Bundle deals increase service depth and "
                          "raise the switching cost for competitors."
            })
        else:
            recs.append({
                "priority": "WATCH",
                "title": "Maintain service quality and monitor",
                "detail": f"Customer has been with the company for {tenure} months and shows low churn risk. "
                          f"No immediate action required — maintain current service quality and re-evaluate at next billing cycle. "
                          f"Consider a loyalty acknowledgement if tenure exceeds 36 months."
            })

    # ── Fallback if no specific recs ─────────────────────────────────────────
    if not recs:
        recs.append({
            "priority": "MEDIUM" if prob >= 0.4 else "WATCH",
            "title": "Schedule proactive check-in",
            "detail": "No single dominant risk factor identified, but churn probability is elevated. "
                      "Assign a customer success rep to schedule a service review call within the next 2 weeks."
        })

    return recs


def gauge_chart(prob, name):
    color = "#ef4444" if prob >= 0.7 else "#f59e0b" if prob >= 0.4 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={
            'text': f"Churn Risk Score<br><span style='font-size:12px;color:#6b7280'>{name}</span>",
            'font': {'color': '#f9fafb', 'size': 14, 'family': 'IBM Plex Sans'}
        },
        number={'suffix': "%", 'font': {'color': '#f9fafb', 'size': 30, 'family': 'IBM Plex Mono'}},
        delta={'reference': 26.58, 'suffix': "%", 'font': {'color': '#6b7280', 'size': 12}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#374151',
                     'tickfont': {'color': '#6b7280', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': '#111827',
            'bordercolor': '#1f2937',
            'steps': [
                {'range': [0, 40],  'color': '#0a2010'},
                {'range': [40, 70], 'color': '#1c1408'},
                {'range': [70, 100], 'color': '#1c0808'},
            ],
            'threshold': {
                'line': {'color': '#9ca3af', 'width': 2},
                'thickness': 0.75,
                'value': 26.58
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f9fafb', 'family': 'IBM Plex Sans'},
        height=270,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig


def risk_bars(tenure, monthly_charges, contract, internet, payment):
    factors = ["Short Tenure", "Contract Type", "Internet Service", "Monthly Charges", "Payment Method"]
    scores = [
        max(0, round(100 - (tenure / 72) * 100, 1)),
        80 if contract == "Month-to-month" else 20 if contract == "One year" else 5,
        70 if internet == "Fiber optic" else 30 if internet == "DSL" else 5,
        round(min(100, ((monthly_charges - 18) / 102) * 100), 1),
        65 if payment == "Electronic check" else 20,
    ]
    colors = ["#ef4444" if s > 60 else "#f59e0b" if s > 30 else "#22c55e" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=factors, orientation='h',
        marker_color=colors,
        text=[f"{s}%" for s in scores], textposition='outside',
        textfont={'color': '#9ca3af', 'size': 11, 'family': 'IBM Plex Mono'}
    ))
    fig.update_layout(
        title={'text': "Risk Factor Breakdown",
               'font': {'color': '#d1d5db', 'size': 13, 'family': 'IBM Plex Sans'}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'range': [0, 115], 'showgrid': False, 'color': '#374151'},
        yaxis={'color': '#9ca3af', 'tickfont': {'size': 11, 'family': 'IBM Plex Sans'}},
        font={'color': '#d1d5db'},
        height=250,
        margin=dict(l=10, r=55, t=40, b=10)
    )
    return fig


# ── HEADER ────────────────────────────────────────────────────────────────────
hl, hc, hr = st.columns([1, 6, 2])
with hl:
    st.markdown("""
    <div style='width:48px;height:48px;background:#00bfa5;border-radius:8px;
    display:flex;align-items:center;justify-content:center;margin-top:6px'>
    <span style='font-size:22px;font-family:monospace;color:#0d1117;font-weight:900'>CG</span>
    </div>""", unsafe_allow_html=True)
with hc:
    st.markdown("""
    <div style='padding:4px 0'>
    <h1 style='margin:0;font-size:22px;font-weight:700;color:#f9fafb;
    letter-spacing:-0.02em;font-family:IBM Plex Sans,sans-serif'>
    ChurnGuard AI</h1>
    <p style='margin:0;color:#6b7280;font-size:12px;letter-spacing:0.04em'>
    TELECOM CUSTOMER INTELLIGENCE &nbsp;|&nbsp; RANDOM FOREST &nbsp;|&nbsp; ROC-AUC: 0.8338
    </p></div>""", unsafe_allow_html=True)
with hr:
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"""
    <div style='text-align:right;padding-top:8px'>
    <span style='color:#6b7280;font-size:11px;font-family:IBM Plex Mono,monospace'>{now}</span><br>
    <span style='background:#00bfa5;color:#0d1117;padding:2px 10px;
    border-radius:3px;font-size:10px;font-weight:700;letter-spacing:0.1em'>LIVE</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 4px'>
    <p style='color:#6b7280;font-size:10px;margin:0;letter-spacing:0.1em;text-transform:uppercase'>
    Customer Profile Input</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    customer_name = st.text_input("Customer Name", placeholder="e.g. Ravi Kumar")
    customer_id   = st.text_input("Customer ID",   placeholder="e.g. CUS-00123")

    st.markdown("---")
    st.markdown("**Account**")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total_charges   = monthly_charges * tenure

    st.markdown("---")
    st.markdown("**Demographics**")
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["No", "Yes"])
    dependents     = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("---")
    st.markdown("**Services**")
    internet         = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    phone            = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protect   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.markdown("**Billing**")
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

    st.markdown("---")
    predict_btn = st.button("Analyze Customer Risk", type="primary", use_container_width=True)
    st.markdown("""<p style='color:#374151;font-size:10px;text-align:center;margin-top:10px;
    font-family:IBM Plex Mono,monospace'>
    Random Forest · IBM Telco · J. Charan Reddy</p>""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Risk Analysis  ",
    "  Analytics  ",
    "  Batch Prediction  ",
    "  Model Info  "
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        inp = build_input(tenure, monthly_charges, total_charges, senior_citizen,
                          partner, dependents, gender, internet, phone, multiple_lines,
                          online_security, online_backup, device_protect, tech_support,
                          streaming_tv, streaming_movies, contract, paperless, payment)
        prediction  = model.predict(inp)[0]
        probability = model.predict_proba(inp)[0][1]

        risk_label, risk_color, risk_class = get_risk_meta(probability)
        name_d  = customer_name if customer_name else "Customer"
        id_d    = customer_id   if customer_id   else "N/A"
        verdict = "CHURN PREDICTED" if prediction == 1 else "RETENTION LIKELY"
        verdict_color = "#ef4444" if prediction == 1 else "#22c55e"

        # ── Risk Banner ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='{risk_class}'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>
                <span style='background:{verdict_color};color:#fff;font-size:10px;font-weight:700;
                padding:3px 10px;border-radius:3px;letter-spacing:0.1em'>{verdict}</span>
                <span style='background:#1f2937;color:#9ca3af;font-size:10px;
                padding:3px 10px;border-radius:3px;letter-spacing:0.06em'>{risk_label}</span>
            </div>
            <p style='color:#f9fafb;font-size:26px;font-weight:700;margin:0;
            font-family:IBM Plex Mono,monospace;letter-spacing:-0.02em'>
                {probability:.1%} <span style='font-size:14px;color:#9ca3af;font-weight:400'>
                churn probability</span>
            </p>
            <p style='color:#9ca3af;font-size:13px;margin:8px 0 0'>
                {name_d} &nbsp;·&nbsp; ID: {id_d}
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Key Metrics ───────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Churn Probability", f"{probability:.1%}",
                  delta=f"{(probability - 0.2658):+.1%} vs avg", delta_color="inverse")
        k2.metric("Risk Level", risk_label)
        k3.metric("Revenue at Risk", f"${monthly_charges:.0f}/mo" if prediction == 1 else "$0")
        k4.metric("Tenure", f"{tenure} months", delta="New" if tenure < 12 else "Established")
        k5.metric("Contract", contract.split()[0])

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────────
        gc, bc = st.columns(2)
        with gc:
            st.plotly_chart(gauge_chart(probability, name_d), use_container_width=True)
        with bc:
            st.plotly_chart(risk_bars(tenure, monthly_charges, contract, internet, payment),
                            use_container_width=True)

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("""<div class='section-header'>
            <h4>Retention Recommendations</h4>
        </div>""", unsafe_allow_html=True)

        recs = build_recommendations(
            probability, tenure, contract, internet, payment,
            monthly_charges, senior_citizen, tech_support,
            online_security, multiple_lines
        )

        priority_labels = {
            "HIGH":   ("HIGH PRIORITY",   "rec-priority-high"),
            "MEDIUM": ("MEDIUM PRIORITY", "rec-priority-medium"),
            "WATCH":  ("MONITOR",         "rec-priority-watch"),
        }

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for rec in recs:
            plabel, pcss = priority_labels[rec["priority"]]
            st.markdown(f"""
            <div class='rec-item'>
                <span class='rec-priority {pcss}'>{plabel}</span>
                <div>
                    <p style='color:#f9fafb;font-size:13px;font-weight:600;margin:0 0 4px'>
                        {rec['title']}</p>
                    <p class='rec-text'>{rec['detail']}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Customer Profile ──────────────────────────────────────────────────
        st.markdown("""<div class='section-header'>
            <h4>Complete Customer Profile</h4>
        </div>""", unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)

        def profile_row(k, v):
            return f"""<div class='profile-row'>
                <span class='profile-key'>{k}</span>
                <span class='profile-value'>{v}</span>
            </div>"""

        with p1:
            st.markdown("<div class='card card-accent-teal'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#00bfa5;font-size:11px;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>Personal</p>",
                        unsafe_allow_html=True)
            for k, v in [("Name", name_d), ("ID", id_d), ("Gender", gender),
                          ("Senior Citizen", senior_citizen), ("Partner", partner), ("Dependents", dependents)]:
                st.markdown(profile_row(k, v), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p2:
            st.markdown("<div class='card card-accent-teal'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#00bfa5;font-size:11px;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>Account</p>",
                        unsafe_allow_html=True)
            for k, v in [("Tenure", f"{tenure} months"), ("Contract", contract),
                          ("Monthly Charges", f"${monthly_charges:.2f}"),
                          ("Total Charges", f"${total_charges:.2f}"),
                          ("Paperless Billing", paperless), ("Payment", payment)]:
                st.markdown(profile_row(k, v), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p3:
            st.markdown("<div class='card card-accent-teal'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#00bfa5;font-size:11px;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>Services</p>",
                        unsafe_allow_html=True)
            for k, v in [("Internet", internet), ("Phone Service", phone),
                          ("Online Security", online_security), ("Tech Support", tech_support),
                          ("Streaming TV", streaming_tv), ("Streaming Movies", streaming_movies)]:
                st.markdown(profile_row(k, v), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Exports ───────────────────────────────────────────────────────────
        st.markdown("---")
        report_text = f"""CHURNGUARD AI — CUSTOMER RISK REPORT
=====================================
Generated  : {datetime.now().strftime("%d %b %Y %H:%M")}
Customer   : {name_d}  |  ID: {id_d}

PREDICTION  : {'CHURN' if prediction == 1 else 'STAY'}
Probability : {probability:.1%}
Risk Level  : {risk_label}

RECOMMENDATIONS
---------------
""" + "\n".join(
            f"[{r['priority']}] {r['title']}\n{r['detail']}\n"
            for r in recs
        ) + f"""
KEY PROFILE
-----------
Contract        : {contract}
Tenure          : {tenure} months
Monthly Charges : ${monthly_charges:.2f}
Internet Service: {internet}
Payment Method  : {payment}

MODEL: Random Forest  |  F1: 0.6225  |  AUC: 0.8338
Built by J. Charan Reddy  |  February 2026
"""
        dl1, dl2, _ = st.columns([1, 1, 3])
        with dl1:
            export_df = pd.DataFrame({
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M")],
                'Customer_Name': [name_d], 'Customer_ID': [id_d],
                'Tenure': [tenure], 'Contract': [contract],
                'Monthly_Charges': [monthly_charges], 'Internet': [internet],
                'Payment': [payment], 'Churn_Probability': [round(probability, 4)],
                'Risk_Segment': [risk_label], 'Prediction': ['CHURN' if prediction == 1 else 'STAY'],
            })
            st.download_button("Export CSV",
                export_df.to_csv(index=False).encode(),
                file_name=f"churn_{name_d.replace(' ', '_')}.csv",
                mime="text/csv", use_container_width=True)
        with dl2:
            st.download_button("Export Report", report_text,
                file_name=f"report_{name_d.replace(' ', '_')}.txt",
                mime="text/plain", use_container_width=True)

    else:
        # ── Welcome State ─────────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center;padding:50px 20px 30px'>
        <div style='width:64px;height:64px;background:#00bfa5;border-radius:10px;
        display:flex;align-items:center;justify-content:center;
        font-size:24px;font-weight:900;color:#0d1117;font-family:monospace;
        margin:0 auto 20px'>CG</div>
        <h2 style='color:#f9fafb;font-size:20px;font-weight:600;margin:0 0 8px'>
        Welcome to ChurnGuard AI</h2>
        <p style='color:#6b7280;font-size:14px;max-width:460px;margin:0 auto'>
        Enter customer details in the sidebar and click
        <span style='color:#00bfa5;font-weight:600'>Analyze Customer Risk</span>
        to get a churn prediction with tailored retention recommendations.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        hw1, hw2, hw3, hw4 = st.columns(4)
        for col, (num, title, desc) in zip([hw1, hw2, hw3, hw4], [
            ("01", "Enter Details",   "Fill customer profile in the sidebar"),
            ("02", "AI Analysis",     "Random Forest evaluates 30 features"),
            ("03", "Risk Score",      "Churn probability and risk segment"),
            ("04", "Take Action",     "Specific recommendations and export"),
        ]):
            col.markdown(f"""
            <div class='card' style='text-align:center'>
            <p style='color:#00bfa5;font-size:22px;font-weight:700;margin:0 0 6px;
            font-family:IBM Plex Mono,monospace'>{num}</p>
            <p style='color:#f9fafb;font-size:13px;font-weight:600;margin:0 0 4px'>{title}</p>
            <p style='color:#6b7280;font-size:12px;margin:0'>{desc}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Training Data",  "7,032 customers")
        m2.metric("Features",       "30")
        m3.metric("F1 Score",       "0.6225")
        m4.metric("ROC-AUC",        "0.8338")
        m5.metric("Recall",         "75.4%")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""<div class='section-header'>
        <h4>Business Analytics — IBM Telco Dataset</h4>
    </div>""", unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Customers",  "7,032")
    a2.metric("Churned",          "1,869", delta="-26.58%", delta_color="inverse")
    a3.metric("Revenue at Risk",  "$35,463/mo")
    a4.metric("Best ROC-AUC",     "0.8338")

    st.markdown("---")

    _chart_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#d1d5db', 'family': 'IBM Plex Sans'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=["Month-to-month", "One year", "Two year"], y=[42.7, 11.3, 2.8],
            marker_color=["#ef4444", "#f59e0b", "#22c55e"],
            text=["42.7%", "11.3%", "2.8%"], textposition="outside",
            textfont={'color': '#9ca3af', 'size': 11}))
        fig.update_layout(title={'text': "Churn Rate by Contract Type",
                                 'font': {'color': '#d1d5db', 'size': 13}},
                          xaxis={'color': '#6b7280'}, yaxis={'color': '#6b7280', 'title': "Churn Rate (%)"},
                          **_chart_layout)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(
            x=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"], y=[47.4, 31.6, 22.1, 8.3],
            marker_color=["#ef4444", "#f59e0b", "#f59e0b", "#22c55e"],
            text=["47.4%", "31.6%", "22.1%", "8.3%"], textposition="outside",
            textfont={'color': '#9ca3af', 'size': 11}))
        fig2.update_layout(title={'text': "Churn Rate by Tenure Group",
                                  'font': {'color': '#d1d5db', 'size': 13}},
                           xaxis={'color': '#6b7280'}, yaxis={'color': '#6b7280', 'title': "Churn Rate (%)"},
                           **_chart_layout)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='F1 Score', x=["LR", "RF", "XGB"], y=[0.6026, 0.6225, 0.6067],
            marker_color='#00bfa5', text=["0.6026", "0.6225", "0.6067"],
            textposition='outside', textfont={'color': '#9ca3af', 'size': 11}))
        fig3.add_trace(go.Bar(name='ROC-AUC', x=["LR", "RF", "XGB"], y=[0.8209, 0.8338, 0.8284],
            marker_color='#0891b2', text=["0.8209", "0.8338", "0.8284"],
            textposition='outside', textfont={'color': '#9ca3af', 'size': 11}))
        fig3.update_layout(title={'text': "Model Performance", 'font': {'color': '#d1d5db', 'size': 13}},
            barmode='group', xaxis={'color': '#6b7280'}, yaxis={'color': '#6b7280', 'range': [0, 1]},
            legend={'font': {'color': '#9ca3af'}}, **_chart_layout)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = go.Figure(go.Pie(
            labels=["Fiber optic 41%", "DSL 19%", "No Internet 7%"],
            values=[41, 19, 7], hole=0.5,
            marker_colors=["#ef4444", "#f59e0b", "#22c55e"],
            textfont={'color': '#d1d5db', 'size': 12}))
        fig4.update_layout(title={'text': "Churn by Internet Service",
                                  'font': {'color': '#d1d5db', 'size': 13}},
                           legend={'font': {'color': '#9ca3af'}}, **_chart_layout)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    rs1, rs2, rs3 = st.columns(3)
    rs1.metric("High Risk",   "303 customers", delta="$22,400 at risk/mo", delta_color="inverse")
    rs2.metric("Medium Risk", "350 customers", delta="$13,063 at risk/mo", delta_color="inverse")
    rs3.metric("Low Risk",    "6,379 customers", delta="Stable")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""<div class='section-header'>
        <h4>Batch Customer Risk Analysis</h4>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='card'>
    <p style='color:#6b7280;margin:0;font-size:13px'>
    Upload a CSV file with customer data to predict churn risk for multiple customers at once.
    File must match the IBM Telco dataset structure.</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded:
        try:
            bdf = pd.read_csv(uploaded)
            st.success(f"{len(bdf)} customers loaded.")
            cust_ids = bdf['customerID'].tolist() if 'customerID' in bdf.columns \
                       else [f"CUST-{i + 1:04d}" for i in range(len(bdf))]
            bdf = bdf.drop(columns=['customerID'], errors='ignore')
            bdf = bdf.drop(columns=['Churn'], errors='ignore')
            bdf['TotalCharges'] = pd.to_numeric(bdf['TotalCharges'], errors='coerce')
            bdf.dropna(inplace=True)
            be = pd.get_dummies(bdf)
            be = be.reindex(columns=feature_cols, fill_value=0)
            be[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
                be[['tenure', 'MonthlyCharges', 'TotalCharges']])
            preds = model.predict(be)
            probs = model.predict_proba(be)[:, 1]
            res = pd.DataFrame({
                'Customer_ID':       cust_ids[:len(preds)],
                'Churn_Probability': np.round(probs, 4),
                'Prediction':        ['CHURN' if p == 1 else 'STAY' for p in preds],
                'Risk_Segment':      ['High Risk' if p >= 0.7 else 'Medium Risk' if p >= 0.4
                                      else 'Low Risk' for p in probs],
                'Monthly_Charges':   bdf['MonthlyCharges'].values[:len(preds)],
                'Tenure':            bdf['tenure'].values[:len(preds)],
                'Contract':          bdf['Contract'].values[:len(preds)],
            })

            bs1, bs2, bs3, bs4 = st.columns(4)
            bs1.metric("Analyzed",    len(res))
            bs2.metric("High Risk",   (res['Risk_Segment'] == 'High Risk').sum())
            bs3.metric("Medium Risk", (res['Risk_Segment'] == 'Medium Risk').sum())
            bs4.metric("Low Risk",    (res['Risk_Segment'] == 'Low Risk').sum())

            st.markdown("---")
            seg_f = st.selectbox("Filter by Risk Segment", ["All", "High Risk", "Medium Risk", "Low Risk"])
            disp  = res[res['Risk_Segment'] == seg_f] if seg_f != "All" else res
            st.dataframe(disp, use_container_width=True, height=400)
            st.download_button("Download Results",
                res.to_csv(index=False).encode(),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:50px 20px'>
        <div style='width:48px;height:48px;background:#1f2937;border-radius:8px;
        display:flex;align-items:center;justify-content:center;
        font-size:20px;margin:0 auto 14px'>
        <span style='color:#6b7280;font-family:monospace;font-weight:700'>CSV</span></div>
        <p style='color:#6b7280;font-size:14px;margin:0'>No file uploaded yet</p>
        <p style='color:#374151;font-size:12px;margin:6px 0 0'>
        Upload a CSV to analyze churn risk across multiple customers at once</p>
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL INFO
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""<div class='section-header'>
        <h4>Model Information & Project Summary</h4>
    </div>""", unsafe_allow_html=True)

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""<div class='card card-accent-teal'>
        <p style='color:#00bfa5;font-size:11px;font-weight:700;
        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>Project Overview</p>
        <p>Random Forest classifier trained on the IBM Telco Customer Churn dataset
        to predict customer churn risk in real time.</p>
        <p><b>Problem:</b> 26.58% of customers churn — $35,463/month in lost revenue.</p>
        <p><b>Solution:</b> ML model combined with SHAP explainability and a live dashboard
        to identify and retain at-risk customers before they leave.</p>
        </div>""", unsafe_allow_html=True)

        perf = pd.DataFrame({
            'Model':   ['Logistic Regression', 'Random Forest (selected)', 'XGBoost'],
            'F1':      [0.6026, 0.6225, 0.6067],
            'ROC-AUC': [0.8209, 0.8338, 0.8284],
            'Recall':  [0.7299, 0.7540, 0.7219],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)

    with i2:
        st.markdown("""<div class='card card-accent-teal'>
        <p style='color:#00bfa5;font-size:11px;font-weight:700;
        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>
        Top Churn Predictors (SHAP)</p>
        <ol style='color:#9ca3af;font-size:13px;line-height:2.2;padding-left:18px'>
        <li><b style='color:#d1d5db'>Tenure</b> — Longer tenure = much lower churn risk</li>
        <li><b style='color:#d1d5db'>Fiber Optic Internet</b> — 41% churn vs 19% DSL</li>
        <li><b style='color:#d1d5db'>Electronic Check Payment</b> — Highest risk payment method</li>
        <li><b style='color:#d1d5db'>Two-Year Contract</b> — Strongest protection against churn</li>
        <li><b style='color:#d1d5db'>Total Charges</b> — Higher total = longer customer = lower risk</li>
        </ol></div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card card-accent-teal' style='margin-top:8px'>
        <p style='color:#00bfa5;font-size:11px;font-weight:700;
        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px'>Technical Stack</p>
        <p style='color:#9ca3af;font-size:13px;line-height:2.2;margin:0'>
        <b style='color:#d1d5db'>Language:</b> Python 3<br>
        <b style='color:#d1d5db'>ML:</b> Scikit-learn · XGBoost · SMOTE<br>
        <b style='color:#d1d5db'>Explainability:</b> SHAP TreeExplainer<br>
        <b style='color:#d1d5db'>Dashboard:</b> Power BI Desktop<br>
        <b style='color:#d1d5db'>App:</b> Streamlit · Plotly<br>
        <b style='color:#d1d5db'>Data:</b> IBM Telco — 7,032 customers, 21 features
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:24px;background:#111827;
    border:1px solid #1f2937;border-radius:10px'>
    <p style='color:#f9fafb;font-size:16px;font-weight:600;margin:0'>J. Charan Reddy</p>
    <p style='color:#6b7280;font-size:12px;margin:6px 0 4px'>Aspiring Data Scientist</p>
    <p style='color:#374151;font-size:11px;margin:0;font-family:IBM Plex Mono,monospace'>
    Python · Machine Learning · Power BI · SHAP · Streamlit · February 2026</p>
    </div>""", unsafe_allow_html=True)
