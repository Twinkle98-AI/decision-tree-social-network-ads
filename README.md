# decision-tree-social-network-ads
End-to-end Machine Learning project using Decision Tree Classifier to predict customer purchase behavior with model persistence using Pickle.
# 📊 Social Network Ads Purchase Prediction (Decision Tree)

This project implements an end-to-end Machine Learning pipeline to predict whether a user will purchase a product based on demographic features using a **Decision Tree Classifier**.

---

## 🔍 Project Overview

The model predicts customer purchase behavior using:
- **Age**
- **Estimated Salary**

The goal is to assist businesses in understanding customer buying patterns and improve targeted marketing strategies.

---

## 🧠 Machine Learning Workflow

✔ Data loading and preprocessing  
✔ Feature selection  
✔ Train–test split  
✔ Feature scaling using StandardScaler  
✔ Model training with Decision Tree (Entropy criterion)  
✔ Model evaluation using accuracy score  
✔ Model persistence using Pickle  

---

## 🗂 Dataset

**Source:** Social Network Ads Dataset  
**Features Used:**
- Age
- Estimated Salary  

**Target Variable:**
- Purchased (0 = No, 1 = Yes)

---

## ⚙️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Pickle  

---

## 📈 Model Performance

- Decision Tree Classifier  
- Criterion: Entropy  
- Max Depth: 5  
- Accuracy achieved on test data  

---

## 💾 Saved Artifacts

- `dt_model.pkl` → Trained Decision Tree model  
- `scaler.pkl` → StandardScaler for feature transformation  

---

## 🚀 How to Run the Project

```bash
pip install numpy pandas scikit-learn
