import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import os
import pdfplumber

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Analyzer",
    page_icon="🏥",
    layout="wide"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.healthy-badge {
    background: #d4edda; color: #155724;
    border-radius: 8px; padding: 4px 10px;
}
.warning-badge {
    background: #fff3cd; color: #856404;
    border-radius: 8px; padding: 4px 10px;
}
.danger-badge {
    background: #f8d7da; color: #721c24;
    border-radius: 8px; padding: 4px 10px;
}
.pdf-section {
    background: #f0f4ff;
    border: 2px dashed #7B9FE0;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state helper ────────────────────────────────────
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# ── Load models ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    models_path = os.path.join(BASE_DIR, '..', 'Models')
    return (
        pickle.load(open(os.path.join(models_path, 'diabetes.pkl'), 'rb')),
        pickle.load(open(os.path.join(models_path, 'heart.pkl'), 'rb')),
        pickle.load(open(os.path.join(models_path, 'kidney.pkl'), 'rb')),
        pickle.load(open(os.path.join(models_path, 'liver.pkl'), 'rb')),
        pickle.load(open(os.path.join(models_path, 'hypertension.pkl'), 'rb')),
        pickle.load(open(os.path.join(models_path, 'scaler.pkl'), 'rb')),
    )

diabetes_model, heart_model, kidney_model, liver_model, hypertension_model, scaler = load_models()

# ── PDF extraction ──────────────────────────────────────────
def extract_value(text, patterns, default=None):
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return default


def extract_patient_info(pdf_text):
    data = {}

    data['age'] = extract_value(pdf_text, [r'age[:\s]+(\d+)'])

    if re.search(r'\b(male|mr)\b', pdf_text, re.I):
        data['gender'] = "Male"
    elif re.search(r'\b(female|ms|mrs)\b', pdf_text, re.I):
        data['gender'] = "Female"

    data['glucose'] = extract_value(pdf_text, [r'glucose[:\s]+([\d.]+)'])
    data['HbA1c'] = extract_value(pdf_text, [r'hba1c[:\s]+([\d.]+)'])
    data['bmi'] = extract_value(pdf_text, [r'bmi[:\s]+([\d.]+)'])

    bp = re.search(r'(\d{2,3})/(\d{2,3})', pdf_text)
    if bp:
        data['sysBP'] = float(bp.group(1))
        data['diaBP'] = float(bp.group(2))

    data['chol'] = extract_value(pdf_text, [r'cholesterol[:\s]+([\d.]+)'])
    data['hemo'] = extract_value(pdf_text, [r'hemoglobin[:\s]+([\d.]+)'])
    data['creatinine'] = extract_value(pdf_text, [r'creatinine[:\s]+([\d.]+)'])
    data['alt'] = extract_value(pdf_text, [r' alt[:\s]+([\d.]+)'])
    data['ast'] = extract_value(pdf_text, [r' ast[:\s]+([\d.]+)'])

    return data


# ── PDF UI ──────────────────────────────────────────────────
st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
st.subheader("📄 Upload PDF")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

extracted = {}

if uploaded_pdf:
    try:
        with pdfplumber.open(io.BytesIO(uploaded_pdf.read())) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

        extracted = extract_patient_info(text)

        if extracted:
            st.success(f"Extracted: {list(extracted.keys())}")
        else:
            st.warning("No data found")
    except Exception as e:
        st.error(e)

st.markdown("</div>", unsafe_allow_html=True)

# ── Helper get value ────────────────────────────────────────
def ex(key, default):
    return extracted.get(key, st.session_state.get(key, default))


# ── INIT SESSION STATE ONCE ─────────────────────────────────
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


# ── INPUT UI ────────────────────────────────────────────────
st.subheader("📋 Patient Input")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, key="age")
    glucose = st.number_input("Glucose", 50, 500, key="glucose")
    HbA1c = st.number_input("HbA1c", 3.0, 15.0, key="HbA1c")
    bmi = st.number_input("BMI", 10.0, 60.0, key="bmi")

with col2:
    sysBP = st.number_input("Sys BP", 80, 250, key="sysBP")
    diaBP = st.number_input("Dia BP", 50, 150, key="diaBP")
    chol = st.number_input("Cholesterol", 100, 600, key="chol")
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, key="hemo")

with col3:
    creatinine = st.number_input("Creatinine", 0.1, 15.0, key="creatinine")
    alt = st.number_input("ALT", 1, 500, key="alt")
    ast = st.number_input("AST", 1, 500, key="ast")

    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")


st.markdown("---")

# ── PREDICTION ──────────────────────────────────────────────
if st.button("🔬 Analyze"):

    gender_male = 1 if st.session_state.gender == "Male" else 0

    diabetes_input = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'HbA1c_level': HbA1c,
        'blood_glucose_level': glucose,
        'gender_Male': gender_male
    }])

    diabetes_pred = diabetes_model.predict(scaler.transform(diabetes_input))[0]

    heart_input = pd.DataFrame([{
        'age': age,
        'trestbps': sysBP,
        'chol': chol,
        'fbs': 1 if glucose > 120 else 0
    }])

    heart_pred = heart_model.predict(heart_input)[0]

    kidney_input = pd.DataFrame([{
        'age': age,
        'bp': sysBP,
        'bgr': glucose,
        'sc': creatinine,
        'hemo': hemo
    }])

    kidney_pred = kidney_model.predict(kidney_input)[0]

    liver_input = pd.DataFrame([{
        'Age': age,
        'ALT': alt,
        'AST': ast,
        'CHOL': chol,
        'CREA': creatinine
    }])

    liver_pred = liver_model.predict(liver_input)[0]

    hypertension_input = pd.DataFrame([{
        'age': age,
        'BMI': bmi,
        'sysBP': sysBP,
        'diaBP': diaBP
    }])

    hypertension_pred = hypertension_model.predict(hypertension_input)[0]

    # ── RESULTS ───────────────────────────────────────────────
    st.subheader("📊 Results")

    def show(name, val):
        if val == 1:
            st.error(f"{name}: At Risk")
        else:
            st.success(f"{name}: Healthy")

    show("Diabetes", diabetes_pred)
    show("Heart", heart_pred)
    show("Kidney", kidney_pred)
    show("Liver", liver_pred)
    show("Hypertension", hypertension_pred)

    # ── CHART ────────────────────────────────────────────────
    fig, ax = plt.subplots()
    vals = [diabetes_pred, heart_pred, kidney_pred, liver_pred, hypertension_pred]
    ax.bar(["D", "H", "K", "L", "Hy"], vals)
    st.pyplot(fig)
