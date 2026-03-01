import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | Telecom Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #0f3460;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="metric-container"] label { color: #888 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 24px !important; font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a2e; border-radius: 10px; padding: 5px; gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #888 !important;
        border-radius: 8px; font-weight: 600; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460, #e94560) !important;
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white !important; border: none; border-radius: 8px;
        font-weight: 700; font-size: 14px; padding: 10px 20px; width: 100%;
    }
    hr { border-color: #0f3460 !important; }
    h1, h2, h3, h4 { color: #ffffff !important; }
    p, li { color: #cccccc !important; }
    .stTextInput > div > div > input {
        background: #1a1a2e !important; color: white !important;
        border: 1px solid #0f3460 !important; border-radius: 8px !important;
    }
    .info-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px;
        padding: 20px; margin: 8px 0;
    }
    .churn-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        border-radius: 12px; padding: 25px; text-align: center;
        border: 2px solid #ff6b6b;
    }
    .churn-medium {
        background: linear-gradient(135deg, #d35400, #e67e22);
        border-radius: 12px; padding: 25px; text-align: center;
        border: 2px solid #ffa07a;
    }
    .churn-low {
        background: linear-gradient(135deg, #1e8449, #2ecc71);
        border-radius: 12px; padding: 25px; text-align: center;
        border: 2px solid #66ff99;
    }
    .section-header {
        background: linear-gradient(90deg, #0f3460, #1a1a2e);
        border-left: 4px solid #e94560;
        padding: 10px 18px; border-radius: 0 8px 8px 0; margin: 15px 0 10px 0;
    }
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

def get_risk(prob):
    if prob >= 0.7:
        return "HIGH RISK","#e74c3c","churn-high","🔴",\
               "Immediate retention action required. Offer contract upgrade with discount."
    elif prob >= 0.4:
        return "MEDIUM RISK","#e67e22","churn-medium","🟠",\
               "Proactive monitoring needed. Schedule check-in call and consider loyalty reward."
    else:
        return "LOW RISK","#2ecc71","churn-low","🟢",\
               "Customer is stable. Maintain service quality and consider upsell opportunities."

def gauge_chart(prob, name):
    color = "#e74c3c" if prob >= 0.7 else "#e67e22" if prob >= 0.4 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={'text': f"Churn Risk Score<br><span style='font-size:13px;color:#aaa'>{name}</span>",
               'font': {'color': 'white', 'size': 16}},
        number={'suffix': "%", 'font': {'color': 'white', 'size': 32}},
        delta={'reference': 26.58, 'suffix': "%", 'font': {'color': '#aaa', 'size': 13}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#aaa', 'tickfont': {'color': '#aaa', 'size': 11}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1a1a2e', 'bordercolor': '#0f3460',
            'steps': [
                {'range': [0, 40],  'color': '#1e3a1e'},
                {'range': [40, 70], 'color': '#3a2a0a'},
                {'range': [70, 100],'color': '#3a0a0a'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.8, 'value': 26.58}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, height=280, margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def risk_bars(tenure, monthly_charges, contract, internet, payment):
    factors = ["Short Tenure","Contract Type","Internet Service","Monthly Charges","Payment Method"]
    scores = [
        max(0, round(100 - (tenure / 72) * 100, 1)),
        80 if contract == "Month-to-month" else 20 if contract == "One year" else 5,
        70 if internet == "Fiber optic" else 30 if internet == "DSL" else 5,
        round(min(100, ((monthly_charges - 18) / 102) * 100), 1),
        65 if payment == "Electronic check" else 20,
    ]
    colors = ["#e74c3c" if s > 60 else "#e67e22" if s > 30 else "#2ecc71" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=factors, orientation='h',
        marker_color=colors,
        text=[f"{s}%" for s in scores], textposition='outside',
        textfont={'color': 'white', 'size': 12}
    ))
    fig.update_layout(
        title={'text': "Risk Factor Breakdown", 'font': {'color': 'white', 'size': 14}},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'range': [0, 115], 'showgrid': False, 'color': '#aaa'},
        yaxis={'color': '#ccc', 'tickfont': {'size': 12}},
        font={'color': 'white'}, height=260, margin=dict(l=10, r=50, t=40, b=10)
    )
    return fig

# ── HEADER ────────────────────────────────────────────────────────────────────
cl, ct, ci = st.columns([1, 5, 2])
with cl:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f3460,#e94560);
    width:60px;height:60px;border-radius:12px;display:flex;
    align-items:center;justify-content:center;font-size:28px;margin-top:5px'>
    🛡️</div>""", unsafe_allow_html=True)
with ct:
    st.markdown("""
    <div style='padding:5px 0'>
    <h1 style='margin:0;font-size:26px;background:linear-gradient(90deg,#fff,#e94560);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    ChurnGuard AI</h1>
    <p style='margin:0;color:#888;font-size:13px'>
    Telecom Customer Intelligence Platform  |  Powered by Random Forest  |  ROC-AUC: 0.8338
    </p></div>""", unsafe_allow_html=True)
with ci:
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"""
    <div style='text-align:right;padding-top:10px'>
    <span style='color:#888;font-size:12px'>🕐 {now}</span><br>
    <span style='background:#0f3460;color:#fff;padding:3px 10px;
    border-radius:20px;font-size:11px;font-weight:700'>LIVE</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:10px 0 5px'>
    <span style='font-size:20px'>👤</span>
    <p style='color:#aaa;font-size:12px;margin:4px 0'>CUSTOMER PROFILE INPUT</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    customer_name = st.text_input("Customer Name", placeholder="e.g. Ravi Kumar")
    customer_id   = st.text_input("Customer ID",   placeholder="e.g. CUS-00123")
    st.markdown("---")
    st.markdown("**📦 Account Information**")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total_charges   = monthly_charges * tenure
    st.markdown("---")
    st.markdown("**🧑 Demographics**")
    gender         = st.selectbox("Gender", ["Male","Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No","Yes"])
    partner        = st.selectbox("Partner", ["No","Yes"])
    dependents     = st.selectbox("Dependents", ["No","Yes"])
    st.markdown("---")
    st.markdown("**📡 Services**")
    internet         = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    phone            = st.selectbox("Phone Service", ["Yes","No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    online_security  = st.selectbox("Online Security", ["No","Yes","No internet service"])
    online_backup    = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    device_protect   = st.selectbox("Device Protection", ["No","Yes","No internet service"])
    tech_support     = st.selectbox("Tech Support", ["No","Yes","No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
    st.markdown("---")
    st.markdown("**💳 Billing**")
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])
    payment   = st.selectbox("Payment Method", [
        "Electronic check","Mailed check",
        "Bank transfer (automatic)","Credit card (automatic)"])
    st.markdown("---")
    predict_btn = st.button("🔍 Analyze Customer Risk", type="primary", use_container_width=True)
    st.markdown("""<p style='color:#555;font-size:10px;text-align:center;margin-top:8px'>
    Model: Random Forest  |  Dataset: IBM Telco<br>
    Built by J. Charan Reddy  |  Feb 2026</p>""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Risk Analysis",
    "📊  Analytics Dashboard",
    "📁  Batch Prediction",
    "ℹ️   Model Info"
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

        risk_label, risk_color, risk_class, risk_icon, advice = get_risk(probability)
        name_d = customer_name if customer_name else "Customer"
        id_d   = customer_id   if customer_id   else "N/A"
        verdict = "⚠️ CHURN PREDICTED" if prediction == 1 else "✅ RETENTION LIKELY"

        st.markdown(f"""
        <div class='{risk_class}'>
        <h1 style='color:white;margin:0;font-size:28px'>{verdict}</h1>
        <h3 style='color:rgba(255,255,255,0.9);margin:8px 0'>{name_d}  ·  ID: {id_d}</h3>
        <h2 style='color:white;margin:0;font-size:36px'>{probability:.1%}
        <span style='font-size:16px;opacity:0.8'> churn probability</span></h2>
        <p style='color:rgba(255,255,255,0.8);margin:8px 0 0;font-size:16px'>{risk_icon} {risk_label}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Churn Probability", f"{probability:.1%}",
                  delta=f"{(probability-0.2658):.1%} vs avg", delta_color="inverse")
        k2.metric("Risk Level", risk_label)
        k3.metric("Revenue at Risk", f"${monthly_charges:.0f}/mo" if prediction==1 else "$0")
        k4.metric("Tenure", f"{tenure} months", delta="New" if tenure<12 else "Established")
        k5.metric("Contract", contract.split()[0])
        st.markdown("---")

        gc, bc = st.columns(2)
        with gc:
            st.plotly_chart(gauge_chart(probability, name_d), use_container_width=True)
        with bc:
            st.plotly_chart(risk_bars(tenure, monthly_charges, contract, internet, payment),
                            use_container_width=True)

        st.markdown(f"""
        <div class='info-card' style='border-left:4px solid {risk_color}'>
        <h4 style='color:{risk_color};margin:0 0 8px'>💡 Recommended Action</h4>
        <p style='color:#ddd;margin:0;font-size:14px'>{advice}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='section-header'>
        <h4 style='margin:0;color:white'>📋 Complete Customer Profile</h4>
        </div>""", unsafe_allow_html=True)

        p1,p2,p3 = st.columns(3)
        with p1:
            st.markdown("""<div class='info-card'>
            <p style='color:#e94560;font-weight:700;margin:0 0 8px'>👤 Personal</p>""",
            unsafe_allow_html=True)
            st.write(f"**Name:** {name_d}"); st.write(f"**ID:** {id_d}")
            st.write(f"**Gender:** {gender}"); st.write(f"**Senior:** {senior_citizen}")
            st.write(f"**Partner:** {partner}"); st.write(f"**Dependents:** {dependents}")
            st.markdown("</div>", unsafe_allow_html=True)
        with p2:
            st.markdown("""<div class='info-card'>
            <p style='color:#e94560;font-weight:700;margin:0 0 8px'>💼 Account</p>""",
            unsafe_allow_html=True)
            st.write(f"**Tenure:** {tenure} months"); st.write(f"**Contract:** {contract}")
            st.write(f"**Monthly:** ${monthly_charges:.2f}"); st.write(f"**Total:** ${total_charges:.2f}")
            st.write(f"**Paperless:** {paperless}"); st.write(f"**Payment:** {payment}")
            st.markdown("</div>", unsafe_allow_html=True)
        with p3:
            st.markdown("""<div class='info-card'>
            <p style='color:#e94560;font-weight:700;margin:0 0 8px'>📡 Services</p>""",
            unsafe_allow_html=True)
            st.write(f"**Internet:** {internet}"); st.write(f"**Phone:** {phone}")
            st.write(f"**Security:** {online_security}"); st.write(f"**Tech Support:** {tech_support}")
            st.write(f"**Streaming TV:** {streaming_tv}"); st.write(f"**Streaming Movies:** {streaming_movies}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        report_text = f"""CHURNGUARD AI — CUSTOMER RISK REPORT
=====================================
Generated  : {datetime.now().strftime("%d %b %Y %H:%M")}
Customer   : {name_d}  |  ID: {id_d}

PREDICTION  : {'CHURN' if prediction==1 else 'STAY'}
Probability : {probability:.1%}
Risk Level  : {risk_label}

RECOMMENDATION: {advice}

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
        dl1, dl2, _ = st.columns([1,1,3])
        with dl1:
            export_df = pd.DataFrame({
                'Timestamp':[datetime.now().strftime("%Y-%m-%d %H:%M")],
                'Customer_Name':[name_d],'Customer_ID':[id_d],
                'Tenure':[tenure],'Contract':[contract],
                'Monthly_Charges':[monthly_charges],'Internet':[internet],
                'Payment':[payment],'Churn_Probability':[round(probability,4)],
                'Risk_Segment':[risk_label],'Prediction':['CHURN' if prediction==1 else 'STAY'],
            })
            st.download_button("📥 Export CSV", export_df.to_csv(index=False).encode(),
                file_name=f"churn_{name_d.replace(' ','_')}.csv", mime="text/csv",
                use_container_width=True)
        with dl2:
            st.download_button("📄 Export Report", report_text,
                file_name=f"report_{name_d.replace(' ','_')}.txt", mime="text/plain",
                use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:50px 20px'>
        <div style='font-size:80px'>🛡️</div>
        <h2 style='color:white;margin:10px 0'>Welcome to ChurnGuard AI</h2>
        <p style='color:#888;font-size:15px;max-width:500px;margin:0 auto'>
        Enter customer details in the sidebar and click
        <b style='color:#e94560'>Analyze Customer Risk</b> to get an AI-powered prediction.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        hw1,hw2,hw3,hw4 = st.columns(4)
        for col,(icon,title,desc) in zip([hw1,hw2,hw3,hw4],[
            ("1️⃣","Enter Details","Fill customer profile in the sidebar"),
            ("2️⃣","AI Analysis","Random Forest analyzes 30 features"),
            ("3️⃣","Risk Score","Churn probability + risk segment assigned"),
            ("4️⃣","Take Action","Get recommendation + export report"),
        ]):
            col.markdown(f"""
            <div class='info-card' style='text-align:center'>
            <div style='font-size:28px'>{icon}</div>
            <h4 style='color:#e94560;margin:8px 0 4px'>{title}</h4>
            <p style='color:#888;font-size:12px;margin:0'>{desc}</p>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Training Data","7,032 customers")
        m2.metric("Features","30")
        m3.metric("F1 Score","0.6225")
        m4.metric("ROC-AUC","0.8338")
        m5.metric("Recall","75.4%")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""<div class='section-header'>
    <h4 style='margin:0;color:white'>📊 Business Analytics — IBM Telco Dataset</h4>
    </div>""", unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Total Customers","7,032")
    a2.metric("Churned","1,869", delta="-26.58%", delta_color="inverse")
    a3.metric("Revenue at Risk","$35,463/mo")
    a4.metric("Best ROC-AUC","0.8338")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=["Month-to-month","One year","Two year"], y=[42.7,11.3,2.8],
            marker_color=["#e74c3c","#e67e22","#2ecc71"],
            text=["42.7%","11.3%","2.8%"], textposition="outside",
            textfont={'color':'white'}))
        fig.update_layout(title={'text':"Churn Rate by Contract Type",'font':{'color':'white'}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'color':'#aaa'}, yaxis={'color':'#aaa','title':"Churn Rate (%)"},
            font={'color':'white'}, height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(
            x=["0-12 mo","13-24 mo","25-48 mo","49-72 mo"], y=[47.4,31.6,22.1,8.3],
            marker_color=["#e74c3c","#e67e22","#f39c12","#2ecc71"],
            text=["47.4%","31.6%","22.1%","8.3%"], textposition="outside",
            textfont={'color':'white'}))
        fig2.update_layout(title={'text':"Churn Rate by Tenure Group",'font':{'color':'white'}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'color':'#aaa'}, yaxis={'color':'#aaa','title':"Churn Rate (%)"},
            font={'color':'white'}, height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig2, use_container_width=True)
    c3,c4 = st.columns(2)
    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='F1 Score', x=["LR","RF","XGB"], y=[0.6026,0.6225,0.6067],
            marker_color='#3498db', text=["0.6026","0.6225","0.6067"],
            textposition='outside', textfont={'color':'white'}))
        fig3.add_trace(go.Bar(name='ROC-AUC', x=["LR","RF","XGB"], y=[0.8209,0.8338,0.8284],
            marker_color='#e94560', text=["0.8209","0.8338","0.8284"],
            textposition='outside', textfont={'color':'white'}))
        fig3.update_layout(title={'text':"Model Performance",'font':{'color':'white'}},
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'color':'#aaa'}, yaxis={'color':'#aaa','range':[0,1]},
            font={'color':'white'}, height=300, legend={'font':{'color':'white'}},
            margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = go.Figure(go.Pie(
            labels=["Fiber optic 41%","DSL 19%","No Internet 7%"],
            values=[41,19,7], hole=0.5,
            marker_colors=["#e74c3c","#e67e22","#2ecc71"],
            textfont={'color':'white','size':12}))
        fig4.update_layout(title={'text':"Churn by Internet Service",'font':{'color':'white'}},
            paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'}, height=300,
            legend={'font':{'color':'white'}}, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")
    rs1,rs2,rs3 = st.columns(3)
    rs1.metric("🔴 High Risk","303 customers", delta="$22,400 at risk/mo", delta_color="inverse")
    rs2.metric("🟠 Medium Risk","350 customers", delta="$13,063 at risk/mo", delta_color="inverse")
    rs3.metric("🟢 Low Risk","6,379 customers", delta="Stable")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""<div class='section-header'>
    <h4 style='margin:0;color:white'>📁 Batch Customer Risk Analysis</h4>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='info-card'>
    <p style='color:#aaa;margin:0;font-size:13px'>
    Upload a CSV file with customer data to predict churn risk for multiple customers at once.
    File must match the IBM Telco dataset structure.</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded:
        try:
            bdf = pd.read_csv(uploaded)
            st.success(f"✅ {len(bdf)} customers loaded!")
            cust_ids = bdf['customerID'].tolist() if 'customerID' in bdf.columns \
                       else [f"CUST-{i+1:04d}" for i in range(len(bdf))]
            bdf = bdf.drop(columns=['customerID'], errors='ignore')
            bdf = bdf.drop(columns=['Churn'], errors='ignore')
            bdf['TotalCharges'] = pd.to_numeric(bdf['TotalCharges'], errors='coerce')
            bdf.dropna(inplace=True)
            be = pd.get_dummies(bdf)
            be = be.reindex(columns=feature_cols, fill_value=0)
            be[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
                be[['tenure','MonthlyCharges','TotalCharges']])
            preds = model.predict(be)
            probs = model.predict_proba(be)[:,1]
            res = pd.DataFrame({
                'Customer_ID': cust_ids[:len(preds)],
                'Churn_Probability': np.round(probs,4),
                'Prediction': ['CHURN' if p==1 else 'STAY' for p in preds],
                'Risk_Segment': ['High Risk' if p>=0.7 else 'Medium Risk' if p>=0.4 else 'Low Risk' for p in probs],
                'Monthly_Charges': bdf['MonthlyCharges'].values[:len(preds)],
                'Tenure': bdf['tenure'].values[:len(preds)],
                'Contract': bdf['Contract'].values[:len(preds)],
            })
            bs1,bs2,bs3,bs4 = st.columns(4)
            bs1.metric("Analyzed", len(res))
            bs2.metric("🔴 High Risk", (res['Risk_Segment']=='High Risk').sum())
            bs3.metric("🟠 Medium Risk", (res['Risk_Segment']=='Medium Risk').sum())
            bs4.metric("🟢 Low Risk", (res['Risk_Segment']=='Low Risk').sum())
            st.markdown("---")
            seg_f = st.selectbox("Filter by Risk Segment", ["All","High Risk","Medium Risk","Low Risk"])
            disp = res[res['Risk_Segment']==seg_f] if seg_f != "All" else res
            st.dataframe(disp, use_container_width=True, height=400)
            st.download_button("📥 Download Results",
                res.to_csv(index=False).encode(),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.markdown("""<div style='text-align:center;padding:40px'>
        <div style='font-size:50px'>📂</div>
        <h4 style='color:#888'>No file uploaded yet</h4>
        <p style='color:#555;font-size:13px'>Upload CSV to analyze churn risk in bulk</p>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL INFO
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""<div class='section-header'>
    <h4 style='margin:0;color:white'>🧠 Model Information & Project Summary</h4>
    </div>""", unsafe_allow_html=True)
    i1,i2 = st.columns(2)
    with i1:
        st.markdown("""<div class='info-card'>
        <h4 style='color:#e94560'>🎯 Project Overview</h4>
        <p>Random Forest classifier trained on IBM Telco Customer Churn dataset
        to predict customer churn risk in real time.</p>
        <p><b>Problem:</b> 26.58% customers churn — $35,463/month lost revenue.</p>
        <p><b>Solution:</b> ML model + SHAP + dashboard to identify and retain
        at-risk customers proactively.</p></div>""", unsafe_allow_html=True)
        perf = pd.DataFrame({
            'Model':['Logistic Regression','Random Forest ★','XGBoost'],
            'F1':[0.6026,0.6225,0.6067],
            'ROC-AUC':[0.8209,0.8338,0.8284],
            'Recall':[0.7299,0.7540,0.7219],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
    with i2:
        st.markdown("""<div class='info-card'>
        <h4 style='color:#e94560'>🔑 Top Churn Predictors (SHAP)</h4>
        <ol style='color:#ccc;font-size:13px;line-height:2.2'>
        <li><b>Tenure</b> — Longer tenure = much lower churn risk</li>
        <li><b>Fiber Optic Internet</b> — 41% churn vs 19% DSL</li>
        <li><b>Electronic Check Payment</b> — Highest risk payment method</li>
        <li><b>Two-Year Contract</b> — Strongest protection against churn</li>
        <li><b>Total Charges</b> — Higher total = longer customer = lower risk</li>
        </ol></div>""", unsafe_allow_html=True)
        st.markdown("""<div class='info-card'>
        <h4 style='color:#e94560'>🛠️ Technical Stack</h4>
        <p style='color:#ccc;font-size:13px;line-height:2'>
        <b>Language:</b> Python 3<br>
        <b>ML:</b> Scikit-learn, XGBoost, SMOTE<br>
        <b>Explainability:</b> SHAP TreeExplainer<br>
        <b>Dashboard:</b> Power BI Desktop<br>
        <b>App:</b> Streamlit + Plotly<br>
        <b>Data:</b> IBM Telco (7,032 customers, 21 features)
        </p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:20px;
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border-radius:12px;border:1px solid #0f3460'>
    <h3 style='color:white;margin:0'>Built by J. Charan Reddy</h3>
    <p style='color:#888;margin:5px 0'>Aspiring Data Scientist</p>
    <p style='color:#555;font-size:12px;margin:0'>
    Python · Machine Learning · Power BI · SHAP · Streamlit · February 2026</p>
    </div>""", unsafe_allow_html=True)