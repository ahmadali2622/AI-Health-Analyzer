# 🏥 AI Health Analyzer

An AI-powered health care web application that analyzes medical reports, predicts disease risk, and provides personalized health recommendations — all in one dashboard.

🔗 **Live Demo:** [ai-health-analyzer.streamlit.app](https://ai-health-analyzer.streamlit.app)

---

## 📌 Overview

AI Health Analyzer uses machine learning to predict the risk of **5 major diseases** from patient health data. It supports both manual input and automatic PDF lab report extraction, making it accessible and practical for real-world use.

---

## ✨ Features

- 🩺 Predicts **5 diseases simultaneously** from a single set of inputs
- 📄 **PDF lab report upload** — auto-extracts patient values (glucose, BP, cholesterol, etc.)
- ✍️ **Manual entry mode** — fill in values directly if no PDF is available
- ✏️ Editable auto-filled fields — review and correct before analysis
- 🔬 Hybrid prediction engine — ML models + clinical threshold rules
- 🏷️ Real-time **health status badges** (Healthy / Borderline / At Risk)
- 📊 Visual result dashboard with bar chart comparison
- 💊 Personalized **diet plan & health recommendations**
- 🧹 Clean, responsive Streamlit UI

---

## 🧠 Diseases Covered

| Disease | Model |
|---------|-------|
| 🩸 Diabetes | Random Forest Classifier |
| ❤️ Heart Disease | Random Forest Classifier |
| 💉 Hypertension | Random Forest Classifier |
| 🫘 Kidney Disease | Random Forest Classifier |
| 🧪 Liver Disease | Random Forest Classifier |

> All models trained on real medical datasets with **85%+ accuracy**

---

## 📁 Project Structure

```
AI-Health-Analyzer/
│
├── App/
│   └── app.py                  # Main Streamlit application
│
├── Dataset/                    # Training datasets (5 diseases)
│
├── Models/                     # Trained ML models (.pkl files)
│   ├── diabetes.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   ├── liver.pkl
│   ├── hypertension.pkl
│   └── scaler.pkl
│
├── Notebook/                   # Jupyter notebooks for training & EDA
│
├── .devcontainer/              # Dev container configuration
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🩺 How It Works

1. **(Optional)** Upload a lab report PDF or fill in values manually
2. Review all health parameters — each shows a live status badge
3. Click **"Analyze Health Report"**
4. App runs 5 ML models + applies clinical threshold rules
5. View risk cards, bar chart comparison, and personalized recommendations

---

## 📄 PDF Auto-Fill Fields

When a PDF is uploaded, the app automatically extracts:

- Age & Gender
- Blood Glucose & HbA1c
- BMI
- Blood Pressure (Systolic / Diastolic)
- Total Cholesterol
- Hemoglobin
- Creatinine
- ALT / AST (Liver enzymes)

> Fields not found in PDF are filled with healthy default values, which you can edit before analysis.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web app framework |
| Scikit-learn | ML model training & prediction |
| Pandas / NumPy | Data handling |
| Matplotlib | Visualizations |
| pdfplumber | PDF text extraction |
| Pickle | Model serialization |

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ahmadali2622/AI-Health-Analyzer.git
cd AI-Health-Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run App/app.py
```

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 👤 Author

**Ahmad Ali**
BS Computer Science — Lahore Leads University
[LinkedIn](https://www.linkedin.com/in/ahmad-ali-117a8a264) | [GitHub](https://github.com/ahmadali2622)
