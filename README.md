# 🏏 IPL Score Analysis & Machine Learning Project (2008-2024)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

---

# 📌 Project Description

This project performs **end-to-end data analysis and machine learning** on **IPL match data from 2008 to 2024**.

The objective is to explore how different phases of the innings affect the **final match score** and to build **predictive models** that estimate match outcomes.

The project demonstrates the **complete data science pipeline**, including:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Data Cleaning
* Statistical Analysis
* Regression Modeling
* Classification Modeling
* Model Evaluation

This project is designed as a **portfolio-ready machine learning project** for data science practice.

---

# 📚 Table of Contents

1. Project Overview
2. Dataset Information
3. Technology Stack
4. Machine Learning Pipeline
5. Project Tasks
6. Visualizations
7. Project Structure
8. Installation
9. How to Run
10. Model Evaluation
11. Future Improvements
12. Author

---

# 📊 Dataset Information

Dataset Used:

**IPL Matches Dataset (2008-2024)**

The dataset contains match statistics including scoring patterns across different overs.

### Key Features

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| First_Innings_Score  | Total runs scored in the first innings  |
| Second_Innings_Score | Total runs scored in the second innings |
| Powerplay_Scores     | Runs scored in first 6 overs            |
| Middle_Overs_Scores  | Runs scored between overs 7-15          |
| Death_Overs_Scores   | Runs scored in final overs              |
| Win_Margin           | Match winning margin                    |

The dataset allows analysis of **how early and middle overs affect final match score**.

---

# 🧰 Technology Stack

### Programming Language

Python

### Libraries Used

* Pandas → Data manipulation
* NumPy → Numerical operations
* Matplotlib → Data visualization
* Seaborn → Statistical plots
* Scikit-Learn → Machine learning models

---

# 🧠 Machine Learning Pipeline

The project follows a structured **Data Science workflow**.

```
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Analysis
   ↓
Regression Modeling
   ↓
Classification Modeling
   ↓
Model Evaluation
```

---

# ⚙️ Project Tasks

## Task 1 — Data Understanding

* Load dataset using Pandas
* Inspect dataset structure
* Identify column names
* Detect quantitative and qualitative data

---

## Task 2 — Exploratory Data Analysis

### Univariate Analysis

Analyzes distribution of a single variable.

Example:

First innings score distribution.

Visualization:

Histogram with KDE.

---

### Bivariate Analysis

Examines relationship between two variables.

Example:

Powerplay runs vs Final score.

Visualization:

Regression plot.

---

### Multivariate Analysis

Analyzes relationships among multiple variables.

Visualization:

Correlation heatmap.

---

# 🧹 Task 3 — Handling Missing Values & Outliers

Steps performed:

* Identify missing values using `isnull()`
* Replace missing values using **median method**
* Detect outliers using **boxplot**

Outliers help identify extreme match outcomes.

---

# 📊 Task 4 — Data Distribution

Statistical analysis performed:

* Standard Deviation
* Skewness
* Kurtosis

Distribution patterns analyzed using histogram.

---

# 🤖 Task 5 — Automated EDA

Reusable Python function created to automate EDA tasks.

Functions used:

* `describe()`
* `info()`
* `isnull()`
* `corr()`

This allows fast analysis of any dataset.

---

# 📈 Task 6 — Regression Analysis

### Target Variable

First_Innings_Score

### Predictor

Powerplay_Scores

Model Used:

Simple Linear Regression.

Analysis performed:

* Covariance
* Correlation

---

# 🧮 Task 7 — Multiple Linear Regression

Model trained using multiple features:

* Powerplay_Scores
* Middle_Overs_Scores
* Death_Overs_Scores

Dataset split into:

* Training set
* Testing set

Used to predict final match score.

---

# ⚖️ Task 8 — Overfitting & Underfitting

Conceptual analysis performed.

* Overfitting → Model memorizes training data
* Underfitting → Model too simple

Training vs testing performance compared.

---

# 🔍 Task 9 — Classification Model

Regression problem converted into classification.

### Classification Rule

```
First_Innings_Score > 170
```

Classes:

| Label | Meaning      |
| ----- | ------------ |
| 1     | High Score   |
| 0     | Normal Score |

Model Used:

Logistic Regression.

Evaluation metrics:

* Accuracy
* Confusion Matrix

---

# 📏 Task 10 — Model Evaluation

Regression performance measured using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

These metrics determine how accurately the model predicts match scores.

---

# 📊 Visualizations Generated

The project generates multiple graphs:

* Score distribution histogram
* Powerplay vs final score regression plot
* Correlation heatmap
* Outlier boxplot
* Regression prediction plot
* Confusion matrix

These visualizations help understand scoring patterns.

---

# 📂 Project Structure

```
IPL-Data-Analysis-Project
│
├── IPL 2008 to 2024.csv
├── use_py.py
├── use_ipynb.ipynb
├── converter.py
├── README.md
```

---

# 💻 Installation

Clone the repository:

```
git clone https://github.com/yourusername/IPL-Data-Analysis.git
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ How to Run the Project

1️⃣ Place dataset in project folder

```
IPL 2008 to 2024.csv
```

2️⃣ Run the Python script

```
python use_py.py
```

3️⃣ Graph windows will open showing visualizations.

Close each graph window to continue the program.

---

# 📊 Example Outputs

The project generates:

* Data distribution graphs
* Correlation heatmaps
* Regression prediction plots
* Confusion matrix
* Model evaluation metrics

---

# 🚀 Future Improvements

Possible improvements for this project:

* Add Random Forest Regression
* Build prediction dashboard
* Create web interface using Streamlit
* Deploy model online
* Add more IPL statistics

---

# 👨‍💻 Author

**Divyesh Kaklotar**

Data Analysis & Machine Learning Project  
📧 **Email:** [pentonick09@gmail.com]  
🔗 **LinkedIn:** [![LinkedIn](https://www.linkedin.com/in/divyesh-kaklotar-p09/)

---

# ⭐ Support

If you like this project:

⭐ Star the repository
📢 Share with others
🚀 Use it for learning data science
