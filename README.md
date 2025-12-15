🎯 CS:GO Round Winner Prediction using Machine Learning
📌 Project Overview

Counter-Strike: Global Offensive (CS:GO) is a popular tactical first-person shooter game where two teams—Terrorists (T) and Counter-Terrorists (CT)—compete to win rounds based on strategy, teamwork, and in-game economy.

In this project, we build a machine learning predictive model that predicts the winner of a round using detailed round-level statistics.

❓ Problem Statement

Predict the round winner (CT or T) based on all available round statistics.

Key Challenge

The dataset contains a large number of features (high dimensionality)

High dimensions can lead to:

Overfitting

Increased computation time

Poor generalization

✅ Solution

To overcome this, we use:

Feature Selection / Feature Extraction

Linear Discriminant Analysis (LDA) to reduce dimensions while preserving class separability.

📂 Dataset Description

Contains round-wise statistics such as:

Kills, deaths, assists

Weapons, grenades, economy

Map information

Team performance metrics

Target variable: Round Winner

🛠️ Technologies & Libraries Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

🔄 Project Workflow
1️⃣ Import Dataset
import pandas as pd
df = pd.read_csv("csgo.csv")

2️⃣ Data Cleaning & Exploratory Data Analysis (EDA)

Checked missing values

Removed duplicates

Analyzed class distribution

Identified categorical & numerical features

3️⃣ Label Encoding

Converted categorical variables into numerical format

pd.get_dummies(df, drop_first=True)

4️⃣ Feature & Target Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

5️⃣ Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

6️⃣ Feature Scaling (StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

7️⃣ Feature Extraction using LDA

Linear Discriminant Analysis (LDA) is used to reduce dimensions while maximizing class separability.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=20)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

🔍 LDA Coefficients (lda.coef_)

Represent weights assigned to each feature

Higher absolute value ⇒ higher importance

Selected top 20 components with coefficient values > 1.25

🤖 Machine Learning Models Implemented
✔ Logistic Regression

Baseline linear classifier

Works well after dimensionality reduction

✔ Decision Tree

Captures non-linear patterns

Prone to overfitting without tuning

✔ Random Forest (Best Performer)

Ensemble technique

Reduces overfitting

Handles high-dimensional data well

✔ SVM & KNN (Optional)

Tested for comparison

Computationally expensive on large features

⚙️ Hyperparameter Tuning

Used GridSearchCV to optimize model performance.

Tuned Parameters:

Logistic Regression: C

Decision Tree:

max_depth

min_samples_split

min_samples_leaf

Random Forest:

n_estimators

max_depth

min_samples_split

📊 Model Performance (After Tuning)
Model	Accuracy
Logistic Regression	~76%
Decision Tree	~80%
Random Forest	~85% (Best)
✅ Final Conclusion

High dimensionality was the main challenge

LDA significantly improved performance

Random Forest with hyperparameter tuning achieved the best accuracy

Feature extraction + ensemble learning proved effective

💡 Key Learnings

Importance of dimensionality reduction in real-world datasets

Practical use of LDA coefficients for feature importance

Hyperparameter tuning improves model generalization

Ensemble models outperform single classifiers