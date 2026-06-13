# 🏥 AI Health Analyzer

AI Health Analyzer is a machine learning–based web application that predicts the risk of multiple diseases using patient health data. It provides instant analysis for **Diabetes, Heart Disease, Kidney Disease, Liver Disease, and Hypertension** in a single dashboard.

Built with **Python, Scikit-learn, and Streamlit**, this project demonstrates an end-to-end machine learning pipeline — from data preprocessing and model training to deployment as an interactive web app.

---

## 🚀 Features

- Predicts 5 major diseases at once from a single set of inputs
- 📄 **Upload a lab report PDF** to auto-extract patient values (age, glucose, BP, cholesterol, etc.)
- ✍️ Manual entry mode — fill in values yourself if no PDF is available
- Editable auto-filled fields — review and correct extracted data before analysis
- Hybrid prediction engine: ML models combined with clinical threshold rules for more reliable results
- Real-time health status badges (Healthy / Borderline / At Risk) for each input
- Visual result dashboard with bar chart comparison
- Personalized health recommendations based on input values
- Clean, responsive Streamlit UI
- Lightweight and easy to deploy

---

## 🧠 Diseases Covered

- 🩸 Diabetes  
- ❤️ Heart Disease  
- 🫘 Kidney Disease  
- 🧪 Liver Disease  
- 💉 Hypertension  

---

## 📄 PDF-Based Auto Fill

Upload a lab report PDF and the app will automatically extract values such as:

- Age & Gender
- Blood Glucose & HbA1c
- BMI
- Blood Pressure (Systolic / Diastolic)
- Total Cholesterol
- Hemoglobin
- Creatinine
- ALT / AST (Liver enzymes)

Any field not found in the PDF is either left blank (Age & Gender — required) or filled with a normal/healthy default value, which you can edit before running the analysis.

---

## 🩺 How It Works

1. (Optional) Upload a lab report PDF, or skip and fill in values manually.
2. Review/edit all health parameters — each shows a live status badge.
3. Click **"Analyze Health Report"**.
4. The app runs 5 ML models, then applies clinical threshold rules to refine results.
5. View results as risk cards, a comparison bar chart, and personalized recommendations.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – web app framework
- **Scikit-learn** – ML models
- **Pandas / NumPy** – data handling
- **Matplotlib** – visualizations
- **pdfplumber** – PDF text extraction

---

## 📁 Project Structure
AI-Health-Analyzer/

├── App/

│   └── app.py

├── Models/

│   ├── diabetes.pkl

│   ├── heart.pkl

│   ├── kidney.pkl

│   ├── liver.pkl

│   ├── hypertension.pkl

│   └── scaler.pkl

├── requirements.txt

└── README.md

---

## ▶️ Running Locally

```bash
git clone https://github.com/yourusername/AI-Health-Analyzer.git
cd AI-Health-Analyzer
pip install -r requirements.txt
streamlit run App/app.py
```

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
