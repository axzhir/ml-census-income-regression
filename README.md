# Machine Learning Project: **Census Income Regression Prep**

## 📁 Overview

This project focuses on preparing the **Census Income** dataset for the modeling phase of a machine learning regression pipeline. The goal is to explore and clean the dataset, engineer useful features, and identify strong predictors for a regression model.

We practice key stages of the machine learning life cycle:

* **Step 2**: Data Cleaning
* **Step 3**: Data Preparation & Exploration

---

## 📝 Objectives

In this project, we will:

### 1️⃣ Build the Data Matrix & Define the ML Problem

* Load the **Census Income** dataset into a Pandas DataFrame
* Define the **label** (target variable) for regression
* Identify potential **features** (independent variables)

### 2️⃣ Clean the Data

* Handle **outliers** by **winsorizing** the label column
* Replace all **missing values** with **column means**

### 3️⃣ Explore the Data

* Identify **two features** with the highest **correlation** with the label
* Create **bivariate plots** to visualize relationships between label and top features

### 4️⃣ Analyze Feature Engineering

* Discuss what **feature engineering techniques** should be applied before modeling (e.g., encoding, scaling, etc.)

---

## 📊 Data

* **Dataset**: Census Income
* **Format**: `.csv` or `.data` file from UCI or your course materials
* **Label (Target)**: *\[Insert what you're predicting, e.g. "Income"]*
* **Features**: A mix of categorical and numerical variables

---

## 🧹 Data Cleaning Techniques

* **Winsorization**: Applied to the label to reduce the influence of outliers
* **Imputation**: Missing values filled using **mean imputation** (simple strategy)

---

## 📈 Exploration & Visualization

* Used **Pearson correlation** to identify top two correlated features
* Built **scatter plots** to show relationships with the label
* Examined patterns and possible non-linearities in feature-target relationships

---

## 🧠 Feature Engineering Plan

* **Categorical Encoding**: Apply one-hot encoding or label encoding
* **Feature Scaling**: Standardize numerical features for models like linear regression
* **Interaction Features**: Consider creating new features based on feature combinations
* **Dimensionality Reduction** (if needed): PCA or feature selection

---

## 🔚 Next Steps

* Finalize feature selection
* Train baseline regression models
* Tune and validate models using cross-validation
* Evaluate performance using MSE, R², etc.

---

## Sample Code
```
import seaborn as sns
import matplotlib.pyplot as plt

# Plot top 2 correlated features with the label
top_features = correlations.index[1:3]  # Exclude the label itself

for feature in top_features:
    sns.scatterplot(data=df, x=feature, y='hours-per-week-winsorized')
    plt.title(f"{feature} vs Hours Per Week")
    plt.show()
```
