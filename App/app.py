import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import os
import pdfplumber

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Analyzer",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# THEME (PRIMARY COLOR #0060A9)
# ─────────────────────────────────────────────
st.markdown("""
<style>

.stApp {
    background-color: #F5F9FC;
}

h1, h2, h3 {
    color: #0060A9 !important;
}

/* Button */
.stButton > button {
    background: #0060A9;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: #004C86;
    transform: scale(1.02);
}

/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox {
    border: 2px solid #0060A9 !important;
    border-radius: 8px !important;
}

/* PDF box */
.pdf-section {
    background: #EAF3FB;
    border: 2px dashed #0060A9;
    border-radius: 12px;
    padding: 16px;
}

/* Badges */
.healthy-badge {
    background: #D6F5E3;
    color: #0B6B3A;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: bold;
}

.warning-badge {
    background: #FFF4D6;
    color: #8A6D00;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: bold;
}

.danger-badge {
    background: #FFD6D6;
    color: #8A1F1F;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT (IMPORTANT FIX)
# ─────────────────────────────────────────────
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

init_state("age", 45)
init_state("glucose", 120)
init_state("HbA1c", 5.5)
init_state("bmi", 25.0)
init_state("sysBP", 120)
init_state("diaBP", 80)
init_state("chol", 180)
init_state("hemo", 13.0)
init_state("creatinine", 0.9)
init_state("alt", 25)
init_state("ast", 25)
init_state("gender", "Male")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    models_path = os.path.join(BASE_DIR, '..', 'Models')

    diabetes_model = pickle.load(open(os.path.join(models_path, 'diabetes.pkl'), 'rb'))
    heart_model = pickle.load(open(os.path.join(models_path, 'heart.pkl'), 'rb'))
    kidney_model = pickle.load(open(os.path.join(models_path, 'kidney.pkl'), 'rb'))
    liver_model = pickle.load(open(os.path.join(models_path, 'liver.pkl'), 'rb'))
    hypertension_model = pickle.load(open(os.path.join(models_path, 'hypertension.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(models_path, 'scaler.pkl'), 'rb'))

    return diabetes_model, heart_model, kidney_model, liver_model, hypertension_model, scaler

diabetes_model, heart_model, kidney_model, liver_model, hypertension_model, scaler = load_models()

# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────
def extract_value(text, patterns):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return None


def extract_patient_info(text):
    data = {}

    data["age"] = extract_value(text, [r'age[:\s]+(\d+)'])
    data["glucose"] = extract_value(text, [r'glucose[:\s]+([\d.]+)'])
    data["HbA1c"] = extract_value(text, [r'hba1c[:\s]+([\d.]+)'])
    data["bmi"] = extract_value(text, [r'bmi[:\s]+([\d.]+)'])
    data["chol"] = extract_value(text, [r'cholesterol[:\s]+([\d.]+)'])
    data["hemo"] = extract_value(text, [r'hemoglobin[:\s]+([\d.]+)'])
    data["creatinine"] = extract_value(text, [r'creatinine[:\s]+([\d.]+)'])
    data["alt"] = extract_value(text, [r'alt[:\s]+([\d.]+)'])
    data["ast"] = extract_value(text, [r'ast[:\s]+([\d.]+)'])

    bp = re.search(r'(\d{2,3})/(\d{2,3})', text)
    if bp:
        data["sysBP"] = float(bp.group(1))
        data["diaBP"] = float(bp.group(2))

    if re.search(r'\b(male|mr)\b', text, re.I):
        data["gender"] = "Male"
    elif re.search(r'\b(female|ms|mrs)\b', text, re.I):
        data["gender"] = "Female"

    return data


# ─────────────────────────────────────────────
# PDF SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
st.subheader("📄 Upload Lab Report PDF")

uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

extracted = {}

if uploaded:
    try:
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

        extracted = extract_patient_info(text)

        if extracted:
            st.success(f"Extracted fields: {list(extracted.keys())}")
        else:
            st.warning("No data found in PDF")
    except Exception as e:
        st.error(e)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS (FIXED SESSION STATE)
# ─────────────────────────────────────────────
st.subheader("📋 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, key="age")
    glucose = st.number_input("Glucose", 50, 500, key="glucose")
    HbA1c = st.number_input("HbA1c", 3.0, 15.0, key="HbA1c")
    bmi = st.number_input("BMI", 10.0, 60.0, key="bmi")

with col2:
    sysBP = st.number_input("Systolic BP", 80, 250, key="sysBP")
    diaBP = st.number_input("Diastolic BP", 50, 150, key="diaBP")
    chol = st.number_input("Cholesterol", 100, 600, key="chol")
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, key="hemo")

with col3:
    creatinine = st.number_input("Creatinine", 0.1, 15.0, key="creatinine")
    alt = st.number_input("ALT", 1, 500, key="alt")
    ast = st.number_input("AST", 1, 500, key="ast")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")

st.markdown("---")

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("🔬 Analyze Health Report"):

    gender_male = 1 if st.session_state.gender == "Male" else 0

    diabetes_input = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "HbA1c_level": HbA1c,
        "blood_glucose_level": glucose,
        "gender_Male": gender_male
    }])

    diabetes = diabetes_model.predict(scaler.transform(diabetes_input))[0]

    heart = heart_model.predict(pd.DataFrame([{
        "age": age,
        "trestbps": sysBP,
        "chol": chol,
        "fbs": 1 if glucose > 120 else 0
    }]))[0]

    kidney = kidney_model.predict(pd.DataFrame([{
        "age": age,
        "bp": sysBP,
        "bgr": glucose,
        "sc": creatinine,
        "hemo": hemo
    }]))[0]

    liver = liver_model.predict(pd.DataFrame([{
        "Age": age,
        "ALT": alt,
        "AST": ast,
        "CHOL": chol,
        "CREA": creatinine
    }]))[0]

    hypertension = hypertension_model.predict(pd.DataFrame([{
        "age": age,
        "BMI": bmi,
        "sysBP": sysBP,
        "diaBP": diaBP
    }]))[0]

    # ─────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────
    st.subheader("📊 Results")

    def show(name, val):
        if val == 1:
            st.error(f"{name}: At Risk")
        else:
            st.success(f"{name}: Healthy")

    show("Diabetes", diabetes)
    show("Heart", heart)
    show("Kidney", kidney)
    show("Liver", liver)
    show("Hypertension", hypertension)

    # ─────────────────────────────────────────────
    # CHART
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.bar(
        ["Diabetes", "Heart", "Kidney", "Liver", "Hypertension"],
        [diabetes, heart, kidney, liver, hypertension]
    )
    st.pyplot(fig)
