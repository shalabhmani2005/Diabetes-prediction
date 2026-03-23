import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# ── Load model, preprocessor, dataset ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    return model, preprocessor


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned_csv"))
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    return df


model, preprocessor = load_artifacts()
df = load_data()

# ── Derive slider ranges from dataset ────────────────────────────────────────
age_min, age_max = int(df["age"].min()), int(df["age"].max())
bmi_min, bmi_max = round(float(df["bmi"].min()), 1), round(float(df["bmi"].max()), 1)
hba1c_min, hba1c_max = round(float(df["HbA1c_level"].min()), 1), round(float(df["HbA1c_level"].max()), 1)
glucose_min, glucose_max = int(df["blood_glucose_level"].min()), int(df["blood_glucose_level"].max())

gender_options = sorted(df["gender"].unique().tolist())
smoking_options = sorted(df["smoking_history"].unique().tolist())

# ── Title ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.write("Adjust the health parameters on the sidebar and get an instant prediction below.")
st.divider()

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Health Parameters")

    gender = st.selectbox("Gender", gender_options, index=gender_options.index("Female") if "Female" in gender_options else 0)

    age = st.slider("Age (years)", min_value=age_min, max_value=age_max, value=30, step=1)

    hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    smoking_history = st.selectbox("Smoking History", smoking_options, index=smoking_options.index("never") if "never" in smoking_options else 0)

    bmi = st.slider("BMI", min_value=bmi_min, max_value=bmi_max, value=25.0, step=0.1, format="%.1f")

    hba1c = st.slider("HbA1c Level", min_value=hba1c_min, max_value=hba1c_max, value=5.5, step=0.1, format="%.1f")

    glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=glucose_min, max_value=glucose_max, value=120, step=1)

# ── Predict ──────────────────────────────────────────────────────────────────
features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]])
transformed = preprocessor.transform(features)
pred = model.predict(transformed)[0]

# ── Display result ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if pred == 0:
        st.success("✅ **Does Not Have Diabetes**")
    else:
        st.error("⚠️ **Has Diabetes**")

# ── Feature summary ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Your Input Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Gender", gender)
m1.metric("Age", f"{age} yrs")

m2.metric("Hypertension", "Yes" if hypertension else "No")
m2.metric("Heart Disease", "Yes" if heart_disease else "No")

m3.metric("Smoking History", smoking_history)
m3.metric("BMI", f"{bmi:.1f}")

m4.metric("HbA1c Level", f"{hba1c:.1f}")
m4.metric("Blood Glucose", f"{glucose} mg/dL")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("🔬 Model: Gradient Boosting Classifier | Preprocessor: ColumnTransformer (OHE + StandardScaler)")
