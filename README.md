# 🚀 Gearbox Predictive Maintenance using Machine Learning

## 📌 Project Overview

This project predicts the health condition of an industrial gearbox using vibration signal analysis and Machine Learning.

The system extracts statistical features from vibration signals stored in `.txt` files and classifies the gearbox condition into:

- ✅ Healthy
- ❌ Broken Tooth

The project compares multiple Machine Learning algorithms and automatically selects the best-performing model.

---

## ✨ Features

- Automatic feature extraction from vibration signals
- Dataset generation
- Data preprocessing
- Training multiple ML models
- Model comparison
- Accuracy evaluation
- Confusion Matrix
- Classification Report
- Automatic Best Model Selection
- Model Saving (.pkl)
- Load saved model
- Predict new vibration files
- Confidence score prediction
- Clean modular code structure

---

## 📂 Project Structure

```
Project_8_Gearbox_Predictive_Maintenance/

│
├── Dataset/
│   ├── Healthy Data/
│   └── BrokenTooth Data/
│
├── Docs/
│
├── Images/
│
├── Models/
│   └── Gearbox_model.pkl
│
├── Outputs/
│   └── Gearbox_features.csv
│
├── Reports/
│
├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── save_model.py
│   ├── load_model.py
│   ├── predict_model.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Machine Learning Models Used

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

The best-performing model is automatically selected based on accuracy.

---

## 📊 Features Extracted

The following statistical features are extracted from each vibration signal:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Peak
- RMS
- Crest Factor

---

## 📈 Dataset

The project uses gearbox vibration signal data consisting of:

- Healthy gearbox signals
- Broken Tooth gearbox signals

Each signal is converted into statistical features before training.

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Joblib

---

## 📦 Installation

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
python main.py
```

---

## 📊 Output

The project provides:

- Dataset Creation
- Model Training
- Model Comparison
- Accuracy Table
- Confusion Matrix
- Classification Report
- Best Model Selection
- Prediction on New Data
- Confidence Score

---

## 🔮 Future Improvements

- Deep Learning (CNN/LSTM)
- Real-time Sensor Integration
- Flask Web Application
- Streamlit Dashboard
- Model Deployment
- Cloud Deployment
- Industrial IoT Integration

---

## 👨‍💻 Author

**Gaurav Kumar**

B.Tech  (Information Technology)

GitHub: *(https://github.com/gauravkumar902704)*

LinkedIn: *(https://www.linkedin.com/in/gaurav-kumar-a756a1278?utm_source=share_via&utm_content=profile&utm_medium=member_android)*

---

⭐ If you like this project, consider giving it a star.