# 🎯 CS:GO Round Winner Prediction

## 📌 Project Overview
This project predicts the winner of a round in Counter-Strike: Global Offensive (CS:GO) using machine learning models based on round-level statistics.

## ❓ Problem Statement
Predict whether Counter-Terrorists (CT) or Terrorists (T) will win a round based on in-game round statistics.

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🔄 Workflow
1. Import dataset
2. Data cleaning & Exploratory Data Analysis (EDA)
3. Label encoding for categorical features
4. Train-test split
5. StandardScaler for feature scaling
6. Feature extraction using Linear Discriminant Analysis (LDA)
7. Model training using:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - SVM (optional)
   - KNN (optional)
8. Hyperparameter tuning using GridSearchCV
9. Model evaluation and comparison

## 📊 Results
Random Forest gave the best performance after feature extraction and hyperparameter tuning.

## ▶ How to Run
1. Create virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

Ensemble models outperform single classifiers
