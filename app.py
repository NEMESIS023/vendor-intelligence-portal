import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import time
import requests
from streamlit_lottie import st_lottie

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Vendor Portal", layout="wide")

# -------------------------------------------------------------
# 🎨 UI STYLE
# -------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}

/* Animated gradient banner */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.cinematic-banner {
    background: linear-gradient(270deg, #020617, #1e293b, #020617);
    background-size: 400% 400%;
    animation: gradientMove 8s ease infinite;
    padding: 25px;
    border-radius: 16px;
    display: flex;
    justify-content: center;
    margin-bottom: 15px;
}

/* Buttons */
div.stButton > button {
    height: 50px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOGIN
# -------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------
def load_model(file):
    return joblib.load(file) if os.path.exists(file) else None

freight_model = load_model("freight_model.pkl")
invoice_model = load_model("invoice_model.pkl")
scaler = load_model("scaler.pkl")

# -------------------------------------------------------------
# LOAD LOTTIE
# -------------------------------------------------------------
def load_lottie(url):
    return requests.get(url).json()

header_lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")

# -------------------------------------------------------------
# 🎬 TOP BANNER (FINAL)
# -------------------------------------------------------------
st.markdown("<div class='cinematic-banner'>", unsafe_allow_html=True)
st_lottie(header_lottie, height=140)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# HEADER (TEXT ONLY)
# -------------------------------------------------------------
st.markdown("""
<h1 style='font-size:36px;'>📊 Vendor Intelligence</h1>""", unsafe_allow_html=True)
st.divider()

# -------------------------------------------------------------
# 🔘 NAVIGATION BUTTONS
# -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "freight"

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("🚚 Freight Cost ", use_container_width=True):
        st.session_state.page = "freight"

with b2:
    if st.button("🚨 Risk Assessment", use_container_width=True):
        st.session_state.page = "risk"

with b3:
    if st.button("📂 Batch Analysis", use_container_width=True):
        st.session_state.page = "batch"

page = st.session_state.page


# =============================================================
# 🚚 FREIGHT
# =============================================================
if page == "freight":

    q = st.number_input("Quantity", value=1000)
    d = st.number_input("Dollars", value=10000.0)

    if st.button("🚀 Predict", use_container_width=True):

        expected = list(freight_model.feature_names_in_)

        input_data = {}
        for col in expected:
            if col.lower() == "quantity":
                input_data[col] = q
            elif col.lower() == "dollars":
                input_data[col] = d

        df = pd.DataFrame([input_data])
        pred = freight_model.predict(df)[0]

        st.metric("Freight Cost", f"${pred:,.2f}")

# =============================================================
# 🚨 RISK
# =============================================================
elif page == "risk":

    iq = st.number_input("Invoice Quantity", value=50)
    fr = st.number_input("Freight", value=1.73)
    idol = st.number_input("Invoice Dollars", value=352.95)
    tiq = st.number_input("Total Items", value=162)
    tid = st.number_input("Total Dollars", value=2476.0)
    tb = st.number_input("Brands", value=5)
    days = st.number_input("Days", value=3)

    if st.button("⚡ Evaluate", use_container_width=True):

        df = pd.DataFrame({
            "invoice_quantity":[iq],
            "invoice_dollars":[idol],
            "Freight":[fr],
            "total_brands":[tb],
            "total_item_quantity":[tiq],
            "days_to_POInvoice":[days],
            "total_item_dollars":[tid]
        })

        scaled = scaler.transform(df)
        pred = invoice_model.predict(scaled)[0]

        if pred == 1:
            st.error("🚨 Risky")
        else:
            st.success("✅ Safe")

# =============================================================
# 📂 BATCH
# =============================================================
else:

    st.subheader("📂 Smart Batch Analysis")

    file = st.file_uploader("Upload CSV")

    if file:
        try:
            df = pd.read_csv(file, encoding="latin1", sep=",", engine="python")
        except:
            st.error("❌ Failed to read CSV")
            st.stop()

        st.dataframe(df.head(100))

        if st.button("📊 Run Analysis", use_container_width=True):

            expected = list(freight_model.feature_names_in_)
            df_cols_lower = {col.lower() for col in df.columns}

            if all(col.lower() in df_cols_lower for col in expected):
                rename_map = {}
                for col in df.columns:
                    for exp in expected:
                        if col.lower() == exp.lower():
                            rename_map[col] = exp

                df_renamed = df.rename(columns=rename_map)
                df["Predicted_Freight"] = freight_model.predict(df_renamed[expected])

            required = [
                "invoice_quantity","invoice_dollars","Freight",
                "total_brands","total_item_quantity",
                "days_to_POInvoice","total_item_dollars"
            ]

            if set(required).issubset(df.columns):
                scaled = scaler.transform(df[required])
                df["Risk_Flag"] = invoice_model.predict(scaled)
            else:
                st.warning("⚠️ Invoice model not applied")

            st.subheader("📊 Data Insights")

            st.dataframe(df.describe())

            num_cols = df.select_dtypes(include=['int64','float64']).columns
            if len(num_cols) > 0:
                sample_df = df.sample(min(len(df), 5000))
                col = st.selectbox("Select column", num_cols)
                st.plotly_chart(px.histogram(sample_df, x=col))

            st.dataframe(df.head(100))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Full Data", csv)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.divider()
st.caption("BCA Final Year Project")