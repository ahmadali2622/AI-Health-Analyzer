import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import os

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Analyzer",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .healthy-badge {
        background: #d4edda; color: #155724;
        border: 1px solid #c3e6cb; border-radius: 8px;
        padding: 4px 12px; font-weight: bold; font-size: 13px;
        display: inline-block; margin-top: 4px;
    }
    .warning-badge {
        background: #fff3cd; color: #856404;
        border: 1px solid #ffc107; border-radius: 8px;
        padding: 4px 12px; font-weight: bold; font-size: 13px;
        display: inline-block; margin-top: 4px;
    }
    .danger-badge {
        background: #f8d7da; color: #721c24;
        border: 1px solid #f5c6cb; border-radius: 8px;
        padding: 4px 12px; font-weight: bold; font-size: 13px;
        display: inline-block; margin-top: 4px;
    }
    .pdf-section {
        background: #f0f4ff;
        border: 2px dashed #7B9FE0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }

    /* Hide number_input +/- step buttons */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        display: none !important;
    }
    div[data-testid="stNumberInput"] > div {
        padding-right: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    models_path = os.path.join(BASE_DIR, '..', 'Models')
    diabetes_model     = pickle.load(open(os.path.join(models_path, 'diabetes.pkl'), 'rb'))
    heart_model        = pickle.load(open(os.path.join(models_path, 'heart.pkl'), 'rb'))
    kidney_model       = pickle.load(open(os.path.join(models_path, 'kidney.pkl'), 'rb'))
    liver_model        = pickle.load(open(os.path.join(models_path, 'liver.pkl'), 'rb'))
    hypertension_model = pickle.load(open(os.path.join(models_path, 'hypertension.pkl'), 'rb'))
    scaler             = pickle.load(open(os.path.join(models_path, 'scaler.pkl'), 'rb'))
    return diabetes_model, heart_model, kidney_model, liver_model, hypertension_model, scaler

diabetes_model, heart_model, kidney_model, liver_model, hypertension_model, scaler = load_models()

# ── Header ───────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#00C9A7;'>🏥 AI Health Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload a lab report PDF or enter values manually for a 5-disease risk assessment</p>", unsafe_allow_html=True)
st.markdown("---")

# ── PDF Upload & Extraction ──────────────────────────────────────
def extract_value(text, patterns, default=None):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    return default

def extract_patient_info(pdf_text):
    data = {}

    data['age'] = extract_value(pdf_text, [
        r'age[:\s]+(\d+)', r'patient age[:\s]+(\d+)', r'(\d+)\s*years?\s*old'
    ])

    if re.search(r'\b(male|man|mr\.)\b', pdf_text, re.IGNORECASE):
        data['gender'] = 'Male'
    elif re.search(r'\b(female|woman|ms\.|mrs\.)\b', pdf_text, re.IGNORECASE):
        data['gender'] = 'Female'

    data['glucose'] = extract_value(pdf_text, [
        r'blood glucose[:\s]+([\d.]+)',
        r'glucose[:\s]+([\d.]+)',
        r'blood sugar[:\s]+([\d.]+)',
        r'fasting glucose[:\s]+([\d.]+)',
    ])

    data['HbA1c'] = extract_value(pdf_text, [
        r'hba1c[:\s]+([\d.]+)',
        r'hb\s*a1c[:\s]+([\d.]+)',
        r'glycated hemoglobin[:\s]+([\d.]+)',
        r'a1c[:\s]+([\d.]+)',
    ])

    data['bmi'] = extract_value(pdf_text, [
        r'bmi[:\s]+([\d.]+)',
        r'body mass index[:\s]+([\d.]+)',
    ])

    bp_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', pdf_text)
    if bp_match:
        data['sysBP'] = float(bp_match.group(1))
        data['diaBP'] = float(bp_match.group(2))
    else:
        data['sysBP'] = extract_value(pdf_text, [
            r'systolic[:\s]+([\d.]+)', r'sys\s*bp[:\s]+([\d.]+)',
        ])
        data['diaBP'] = extract_value(pdf_text, [
            r'diastolic[:\s]+([\d.]+)', r'dia\s*bp[:\s]+([\d.]+)',
        ])

    data['chol'] = extract_value(pdf_text, [
        r'total cholesterol[:\s]+([\d.]+)',
        r'cholesterol[:\s]+([\d.]+)',
        r'chol[:\s]+([\d.]+)',
    ])

    data['hemo'] = extract_value(pdf_text, [
        r'hemoglobin[:\s]+([\d.]+)',
        r'haemoglobin[:\s]+([\d.]+)',
        r'\bhgb[:\s]+([\d.]+)',
        r'\bhb[:\s]+([\d.]+)',
    ])

    data['creatinine'] = extract_value(pdf_text, [
        r'creatinine[:\s]+([\d.]+)',
        r'serum creatinine[:\s]+([\d.]+)',
    ])

    data['alt'] = extract_value(pdf_text, [
        r'\balt[:\s]+([\d.]+)',
        r'alanine aminotransferase[:\s]+([\d.]+)',
        r'sgpt[:\s]+([\d.]+)',
    ])
    data['ast'] = extract_value(pdf_text, [
        r'\bast[:\s]+([\d.]+)',
        r'aspartate aminotransferase[:\s]+([\d.]+)',
        r'sgot[:\s]+([\d.]+)',
    ])

    return data

# ── PDF Section ───────────────────────────────────────────────────
st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
st.markdown("### 📄 Upload Lab Report PDF *(optional)*")
st.caption("Upload a PDF lab report to auto-fill patient fields. Age and Gender must be entered manually if not found in the PDF. Other missing fields will default to normal/healthy values, which you can edit.")

uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

extracted = {}
if uploaded_pdf is not None:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded_pdf.read())) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        if full_text.strip():
            extracted = extract_patient_info(full_text)
            found = [k for k, v in extracted.items() if v is not None]
            if found:
                st.success(f"✅ Extracted {len(found)} field(s): {', '.join(found)}. Review and edit below.")
            else:
                st.warning("⚠️ PDF was read but no recognizable health values were found. Please fill in manually.")
        else:
            st.warning("⚠️ Could not read text from this PDF (may be a scanned image). Please fill in manually.")
    except ImportError:
        st.error("❌ `pdfplumber` not installed. Run: `pip install pdfplumber` then restart.")
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────
field_keys = ['age', 'glucose', 'HbA1c', 'bmi', 'sysBP', 'diaBP',
               'chol', 'hemo', 'creatinine', 'alt', 'ast', 'gender']

# Fields that must be int vs float (must match number_input min/max types below)
int_fields = {'age', 'glucose', 'sysBP', 'diaBP', 'chol', 'alt', 'ast'}
float_fields = {'HbA1c', 'bmi', 'hemo', 'creatinine'}

# Normal/healthy default values used when a field isn't found in the PDF.
# age and gender are intentionally excluded — they're required from the user.
normal_defaults = {
    'glucose': 100,
    'HbA1c': 5.4,
    'bmi': 22.0,
    'sysBP': 118,
    'diaBP': 78,
    'chol': 180,
    'hemo': 14.0,
    'creatinine': 0.9,
    'alt': 25,
    'ast': 25,
}

def cast_field(k, v):
    if v is None:
        return None
    if k in int_fields:
        return int(v)
    elif k in float_fields:
        return float(v)
    return v  # gender, etc.

for k in field_keys:
    if k not in st.session_state:
        val = cast_field(k, extracted.get(k))
        if val is None and k in normal_defaults:
            val = cast_field(k, normal_defaults[k])
        st.session_state[k] = val  # age/gender stay None if not found in PDF

# Track the last processed PDF so re-uploading a NEW pdf overwrites fields,
# but rerunning with the SAME pdf doesn't keep resetting your edits.
last_pdf_name = st.session_state.get('_last_pdf_name')
current_pdf_name = uploaded_pdf.name if uploaded_pdf is not None else None

if current_pdf_name is not None and current_pdf_name != last_pdf_name:
    for k in field_keys:
        v = extracted.get(k)
        if v is not None:
            st.session_state[k] = cast_field(k, v)
        elif k in normal_defaults:
            # Not found in this PDF — reset to normal default
            st.session_state[k] = cast_field(k, normal_defaults[k])
        else:
            # age / gender — clear so user must enter manually
            st.session_state[k] = None
    st.session_state['_last_pdf_name'] = current_pdf_name
elif current_pdf_name is None:
    st.session_state['_last_pdf_name'] = None

# ── Status badge helper ──────────────────────────────────────────
def status_badge(value, healthy_max, warning_max=None, unit="", low_warning=None):
    if value is None:
        return ""
    if low_warning is not None and value < low_warning:
        return f'<span class="warning-badge">⚠️ Low ({value}{unit})</span>'
    elif value <= healthy_max:
        return f'<span class="healthy-badge">✅ Healthy ({value}{unit})</span>'
    elif warning_max is not None and value <= warning_max:
        return f'<span class="warning-badge">⚠️ Borderline ({value}{unit})</span>'
    else:
        return f'<span class="danger-badge">🔴 At Risk ({value}{unit})</span>'

# ── Input Section ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Patient Information")
st.caption("Fields auto-filled from PDF are editable — Age and Gender are required.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state['age'], key='age')
    glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=500, value=st.session_state['glucose'], key='glucose')
    st.markdown(status_badge(glucose, 99, 125, " mg/dL"), unsafe_allow_html=True)

    HbA1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=st.session_state['HbA1c'], key='HbA1c')
    st.markdown(status_badge(HbA1c, 5.6, 6.4, "%"), unsafe_allow_html=True)

    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=st.session_state['bmi'], key='bmi')
    st.markdown(status_badge(bmi, 24.9, 29.9, ""), unsafe_allow_html=True)

with col2:
    sysBP = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=st.session_state['sysBP'], key='sysBP')
    st.markdown(status_badge(sysBP, 120, 130, " mmHg"), unsafe_allow_html=True)

    diaBP = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=150, value=st.session_state['diaBP'], key='diaBP')
    st.markdown(status_badge(diaBP, 80, 89, " mmHg"), unsafe_allow_html=True)

    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=st.session_state['chol'], key='chol')
    st.markdown(status_badge(chol, 199, 239, " mg/dL"), unsafe_allow_html=True)

    hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=st.session_state['hemo'], key='hemo')
    st.markdown(status_badge(hemo, 17.0, None, " g/dL", low_warning=11.0), unsafe_allow_html=True)

with col3:
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=st.session_state['creatinine'], key='creatinine')
    st.markdown(status_badge(creatinine, 1.2, 1.5, " mg/dL"), unsafe_allow_html=True)

    alt = st.number_input("ALT (U/L)", min_value=1, max_value=500, value=st.session_state['alt'], key='alt')
    st.markdown(status_badge(alt, 40, 56, " U/L"), unsafe_allow_html=True)

    ast = st.number_input("AST (U/L)", min_value=1, max_value=500, value=st.session_state['ast'], key='ast')
    st.markdown(status_badge(ast, 40, 55, " U/L"), unsafe_allow_html=True)

    gender_options = ["Male", "Female"]
    gender_val = st.session_state['gender']
    gender_index = gender_options.index(gender_val) if gender_val in gender_options else None
    gender = st.selectbox("Gender", gender_options, index=gender_index, key='gender', placeholder="Select gender")

st.markdown("---")

# ── Predict Button ───────────────────────────────────────────────
if st.button("🔬 Analyze Health Report", use_container_width=True):

    # ── Validate required fields (age, gender) ─────────────────────
    missing = [k for k in field_keys if st.session_state[k] is None]
    if missing:
        st.error(f"⚠️ Please fill in: {', '.join(missing)} — these could not be detected from the PDF.")
        st.stop()

    gender_male = 1 if gender == "Male" else 0

    # ── Diabetes ──
    diabetes_input = pd.DataFrame([{
        'age': age, 'hypertension': 1 if sysBP > 130 else 0,
        'heart_disease': 0, 'bmi': bmi,
        'HbA1c_level': HbA1c, 'blood_glucose_level': glucose,
        'gender_Male': gender_male, 'gender_Other': 0,
        'smoking_history_current': 0, 'smoking_history_ever': 0,
        'smoking_history_former': 0, 'smoking_history_never': 1,
        'smoking_history_not current': 0,
    }])
    diabetes_input_scaled = scaler.transform(diabetes_input)
    dibeties = diabetes_model.predict(diabetes_input_scaled)[0]

    # ── Heart ──
    heart_input = pd.DataFrame([{
        'age': age, 'sex': gender_male, 'cp': 0,
        'trestbps': sysBP, 'chol': chol,
        'fbs': 1 if glucose > 120 else 0,
        'restecg': 0, 'thalach': 75, 'exang': 0,
        'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
    }])
    heart = heart_model.predict(heart_input)[0]

    # ── Kidney ──
    kidney_input = pd.DataFrame([{
        'age': age, 'bp': sysBP, 'sg': 1.020,
        'al': 0, 'su': 0, 'bgr': glucose,
        'bu': 20, 'sc': creatinine,
        'sod': 138, 'pot': 4.2, 'hemo': hemo,
        'pcv': 40.0, 'wc': 7200.0, 'rc': 4.5,
        'rbc_normal': 1, 'pc_normal': 1,
        'pcc_present': 0, 'ba_present': 0,
        'htn_yes': 1 if sysBP > 130 else 0,
        'dm_yes': 1 if glucose > 140 else 0,
        'cad_no': 1, 'cad_yes': 0,
        'appet_poor': 0, 'pe_yes': 0, 'ane_yes': 0
    }])
    kidney_input = kidney_input.reindex(columns=kidney_model.feature_names_in_, fill_value=0)
    kidney = kidney_model.predict(kidney_input)[0]

    # ── Liver ──
    liver_input = pd.DataFrame([{
        'Age': age, 'Sex': gender_male,
        'ALB': 4.0, 'ALP': 70, 'ALT': alt,
        'AST': ast, 'BIL': 0.8, 'CHE': 8.0,
        'CHOL': chol, 'CREA': creatinine,
        'GGT': 30, 'PROT': 7.0
    }])
    liver_input = liver_input.reindex(columns=liver_model.feature_names_in_, fill_value=0)
    liver = liver_model.predict(liver_input)[0]

    # ── Hypertension ──
    hypertension_input = pd.DataFrame([{
        'age': age, 'BMI': bmi,
        'sysBP': sysBP, 'diaBP': diaBP,
        'glucose': glucose, 'totChol': chol
    }])
    hypertension = hypertension_model.predict(hypertension_input)[0]

    # ── Clinical override rules ───────────────────────────────────
    def clinical_diabetes(model_result):
        if glucose >= 126 or HbA1c >= 6.5:
            return 1
        elif glucose < 100 and HbA1c < 5.7:
            return 0
        else:
            return model_result

    def clinical_heart(model_result):
        if chol > 240 or sysBP > 140:
            return 1
        elif chol < 200 and sysBP < 120:
            return 0
        else:
            return model_result

    def clinical_kidney(model_result):
        if creatinine > 1.2:
            return 1
        elif creatinine < 0.9:
            return 0
        else:
            return model_result

    def clinical_liver(model_result):
        if alt > 56 or ast > 40:
            return 1
        elif alt < 25 and ast < 25:
            return 0
        else:
            return model_result

    def clinical_hypertension(model_result):
        if sysBP > 130 or diaBP > 80:
            return 1
        elif sysBP < 120 and diaBP < 80:
            return 0
        else:
            return model_result

    dibeties_final     = clinical_diabetes(dibeties)
    heart_final        = clinical_heart(heart)
    kidney_final       = clinical_kidney(kidney)
    liver_final        = clinical_liver(liver)
    hypertension_final = clinical_hypertension(hypertension)

    # ── Results ──────────────────────────────────────────────────
    st.markdown("## 📊 Health Report Results")
    c1, c2, c3, c4, c5 = st.columns(5)

    def show_result(col, name, result, icon):
        with col:
            if result == 1:
                st.error(f"{icon}\n\n**{name}**\n\n⚠️ At Risk")
            else:
                st.success(f"{icon}\n\n**{name}**\n\n✅ Healthy")

    show_result(c1, "Diabetes",       dibeties_final,     "🩸")
    show_result(c2, "Heart Disease",  heart_final,        "❤️")
    show_result(c3, "Kidney Disease", kidney_final,       "🫘")
    show_result(c4, "Liver Disease",  liver_final,        "🫀")
    show_result(c5, "Hypertension",   hypertension_final, "💉")

    # ── Bar Chart ────────────────────────────────────────────────
    st.markdown("---")
    labels = ["Diabetes", "Heart", "Kidney", "Liver", "Hypertension"]
    values = [dibeties_final, heart_final, kidney_final, liver_final, hypertension_final]
    colors = ["#E74C3C" if v == 1 else "#2ECC71" for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                "At Risk" if val == 1 else "Healthy",
                ha='center', fontsize=10, fontweight='bold',
                color="#E74C3C" if val == 1 else "#2ECC71")
    ax.set_title("Disease Risk Analysis", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1], ["Healthy", "At Risk"])
    st.pyplot(fig)

    # ── Recommendations ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 💊 Recommendations")

    if glucose > 200:
        st.warning("🔴 Very High Glucose — Immediate medical consultation recommended")
    elif glucose > 140:
        st.warning("🔴 Reduce Sugar Intake & HbA1c test Recommended")
    elif glucose > 100:
        st.info("🟡 Borderline Glucose — Monitor diet and sugar intake")
    else:
        st.success("✅ Glucose levels are Normal")

    if sysBP > 140:
        st.warning("🔴 Reduce Salt Intake & Check BP Regularly")
    elif sysBP > 120:
        st.info("🟡 Slightly Elevated BP — Reduce stress & salt")
    else:
        st.success("✅ Blood Pressure is Normal")

    if chol > 240:
        st.warning("🔴 Avoid Oily Food & Lipid Profile Test Recommended")
    elif chol > 200:
        st.info("🟡 Borderline Cholesterol — Reduce fried food")
    else:
        st.success("✅ Cholesterol is Normal")

    if HbA1c > 6.5:
        st.warning("🔴 High HbA1c — Consult a diabetologist")
    elif HbA1c > 5.7:
        st.info("🟡 Pre-diabetic HbA1c range — Monitor carefully")
    else:
        st.success("✅ HbA1c is Normal")

    if creatinine > 1.2:
        st.warning("🔴 High Creatinine — Kidney function test recommended")
    else:
        st.success("✅ Creatinine is Normal")

    if alt > 56 or ast > 40:
        st.warning("🔴 Elevated Liver Enzymes — Liver function test recommended")
    else:
        st.success("✅ Liver Enzymes are Normal")
