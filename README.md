```md
# 🏥 AI Health Analyzer

AI Health Analyzer is a machine learning–based web application that predicts the risk of multiple diseases using patient health data. It provides instant analysis for **Diabetes, Heart Disease, Kidney Disease, Liver Disease, and Hypertension** in a single dashboard.

Built with **Python, Scikit-learn, and Streamlit**, this project demonstrates an end-to-end machine learning pipeline from training to deployment.

---

## 🚀 Features

- Predicts 5 major diseases at once
- Clean and interactive Streamlit UI
- Real-time health risk analysis
- Visual result dashboard with charts
- Basic health recommendations based on inputs
- Lightweight and easy to deploy

---

## 🧠 Diseases Covered

- 🩸 Diabetes  
- ❤️ Heart Disease  
- 🫘 Kidney Disease  
- 🫀 Liver Disease  
- 💉 Hypertension  

---

## 📁 Project Structure

```

AI-Health-Analyzer/
│
├── notebooks/
│     └── model_training.ipynb
│
├── app/
│     └── streamlit_app.py
│
├── models/
│     ├── diabetes.pkl
│     ├── heart.pkl
│     ├── kidney.pkl
│     ├── liver.pkl
│     ├── hypertension.pkl
│     └── scaler.pkl
│
├── data/
│     └── dataset.csv
│
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ahmadali2622/AI-Health-Analyzer.git
cd AI-Health-Analyzer
````

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. User enters health parameters (age, glucose, BP, cholesterol, etc.)
2. Pre-trained ML models process the input
3. Each disease model predicts risk (0 = Healthy, 1 = At Risk)
4. Results are displayed with:

   * Risk labels
   * Color indicators
   * Bar chart visualization
   * Health recommendations

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Streamlit

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
Pillow
```

---

## 🌐 Deployment

You can deploy this project using **Streamlit Cloud**:

1. Push project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select:

   * Main file: `app/streamlit_app.py`
5. Click **Deploy 🚀**

---

## 📊 Example Use Case

This system can be used for:

* Quick health risk screening
* Educational ML projects
* AI healthcare demonstrations
* Portfolio projects for internships/jobs

---

## 🔮 Future Improvements

* Add medical report OCR scanning
* Add patient history tracking
* Improve model accuracy with deep learning
* Add authentication system
* Add downloadable health report (PDF)

---

## 👨‍💻 Author

**Ahmed Ali**
BS Computer Science Student
Interested in AI, Machine Learning, and Full Stack Development

---

## ⭐ Support

If you like this project:

* Give it a ⭐ on GitHub
* Share it with others
* Use it in your portfolio

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a medical diagnosis tool.

```
```