import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import warnings

# --- SETUP ---
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

def next_task(task_no):
    print(f"\n--- Task {task_no} Completed ! ---")
    input("PRESS ENTER FOR NEXT TASK...")

# --- TASK 1: Libraries & Environment ---
print(">>> Task 1: Loading libraries...")
print("Pandas, Numpy, Matplotlib, Seaborn, aur Sklearn are ready !")
next_task(1)

# --- TASK 2: Load and Inspect Dataset ---
print("\n>>> Task 2: Dataset Loading...")
try:
    df = pd.read_csv('IPL 2008 to 2024.csv')
    print("Dataset Loaded Successfully!")
    
    print("\n2.2.1 First 5 Rows:")
    print(df.head())
    print("\n2.2.2 Last 5 Rows:")
    print(df.tail())
    print("\nShape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    quant_data = df.select_dtypes(include=['number']).columns.tolist()
    qual_data = df.select_dtypes(exclude=['number']).columns.tolist()
    print("\n2.3.1 Quantitative (Numbers):", quant_data)
    print("2.3.2 Qualitative (Text):", qual_data)
except FileNotFoundError:
    print("Error: 'IPL 2008 to 2024.csv' file nahi mili! File ko isi folder mein rakhein.")
    exit()
next_task(2)

# --- TASK 3: Exploratory Data Analysis (Graphs) ---
print("\n>>> Task 3: Visualizations (Graphs)...")

# 3.1 Univariate Analysis
plt.figure(figsize=(8, 5))
ax = sns.histplot(df['First_Innings_Score'], kde=True, color='blue', bins=20)
avg_score = df['First_Innings_Score'].mean()
plt.axvline(avg_score, color='red', linestyle='--', linewidth=2)
plt.text(avg_score + 2, ax.get_ylim()[1]*0.9, f'Average Score: {avg_score:.0f}', color='red', fontsize=12, weight='bold')
plt.title('3.1 Univariate: How many runs are scored in each IPL match ?', fontsize=14)
plt.xlabel('First Innings Final Score', fontsize=12)
plt.ylabel('Count of matches', fontsize=12)
print("Graph 1 is visible... close it to proceed.")
plt.show()

# 3.2 Bivariate Analysis
plt.figure(figsize=(8, 5))
sns.regplot(x='Powerplay_Scores', y='First_Innings_Score', data=df, scatter_kws={'color': 'green', 'alpha': 0.6}, line_kws={'color': 'red', 'linewidth': 2})
plt.title('3.2 Bivariate: How do powerplay runs affect the final score ?', fontsize=14)
plt.xlabel('Powerplay Runs (The first 6 overs)', fontsize=12)
plt.ylabel('Final Score (After 20 years)', fontsize=12)
plt.text(df['Powerplay_Scores'].min(), df['First_Innings_Score'].max() - 10, "Red line going up = Good powerplay, good score", color='red', fontsize=11, weight='bold')
print("Graph 2 is visible... close it.")
plt.show()

# 3.3 Multivariate Analysis
plt.figure(figsize=(10, 6))
numeric_cols = df[['First_Innings_Score', 'Powerplay_Scores', 'Middle_Overs_Scores', 'Death_Overs_Scores']]
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, annot_kws={"size": 13, "weight": "bold"})
plt.title('3.3 Multivariate: Correlation between scores', fontsize=15, weight='bold', pad=20)
plt.figtext(0.5, -0.05,"Red = strong connection (e.g. 0.83)\nMeaning, if more runs are scored in the middle overs, the final score will definitely be big!", ha="center", fontsize=12,bbox={"facecolor":"yellow", "alpha":0.4, "pad":8},color="black", weight="bold")
print("Graph 3 (Hetmap) I see... Close it.")
plt.show()
next_task(3)

# --- TASK 4: Data Cleaning ---
print("\n>>> Task 4: Identify & Handle Missing Values/Outliers...")
print("Null Values:\n", df.isnull().sum())

df['Win_Margin'] = df['Win_Margin'].fillna(df['Win_Margin'].median())
df['Second_Innings_Score'] = df['Second_Innings_Score'].fillna(df['Second_Innings_Score'].median())

plt.figure(figsize=(11, 5))
sns.boxplot(x=df['Win_Margin'], color='orange')
plt.title('4.3 Outliers: By how many runs did the team win ?', fontsize=14)
plt.xlabel('Winning margin (Win margin in runs/wickets))', fontsize=12)
plt.annotate('These are the outliers (win by 100+ runs)',xy=(100, 0), xytext=(80, 0.3),arrowprops=dict(facecolor='black', shrink=0.05),fontsize=12, color='black', weight='bold')
print("Boxplot (Outliers) is visible... close it.")
plt.show()
next_task(4)

# --- TASK 5: Statistical Measures ---
print("\n>>> Task 5: Distribution & Stats...")
plt.figure(figsize=(8, 5))
sns.histplot(df['First_Innings_Score'], kde=True, color='red')
mean_val = df['First_Innings_Score'].mean()
median_val = df['First_Innings_Score'].median()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
plt.axvline(median_val, color='blue', linewidth=2, label=f'Median: {median_val:.1f}')
plt.title('5.1 Data Distribution (Normal vs Skewed)')
plt.legend()
plt.show()

std_dev = df['First_Innings_Score'].std()
kurt = df['First_Innings_Score'].kurtosis()
skew = df['First_Innings_Score'].skew()
print(f"5.2.1 Standard Deviation: {std_dev:.2f}")
print(f"5.2.2 Kurtosis: {kurt:.2f}")
print(f"Skewness: {skew:.2f}")
next_task(5)

# --- TASK 6: Reusable Function ---
print("\n>>> Task 6: Reusable Python Function Test...")
def auto_eda_function(data):
    print("--- 6.1.2 Info ---")
    data.info()
    print("\n--- 6.1.1 Describe ---")
    print(data.describe())
    print("\n--- 6.1.3 Isnull ---")
    print(data.isnull().sum())
    print("\n--- 6.1.4 Corr ---")
    print(data.select_dtypes(include='number').corr())

auto_eda_function(df)
next_task(6)

# --- TASK 7: Simple Linear Regression ---
print("\n>>> Task 7: Simple Linear Regression...")
X = df[['Powerplay_Scores']]
y = df['First_Innings_Score']
model = LinearRegression()
model.fit(X, y)
pred_y = model.predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Powerplay_Scores'], y=y, color='lightblue', label='Actual Score')
plt.plot(df['Powerplay_Scores'], pred_y, color='red', linewidth=2.5, label='Prediction')
plt.xlabel('Powerplay Runs')
plt.ylabel('Final Score')
plt.legend()
plt.show()

covar = np.cov(df['Powerplay_Scores'], df['First_Innings_Score'])[0][1]
corre = df['Powerplay_Scores'].corr(df['First_Innings_Score'])
print(f"7.3.1 Covariance: {covar:.2f}")
print(f"7.3.2 Correlation: {corre:.2f}")
next_task(7)

# --- TASK 8: Multiple Linear Regression ---
print("\n>>> Task 8: Multiple Linear Regression...")
X_multi = df[['Powerplay_Scores', 'Middle_Overs_Scores', 'Death_Overs_Scores']]
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

multi_reg = LinearRegression()
multi_reg.fit(X_train, y_train)
y_pred_multi = multi_reg.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_multi, color='green', alpha=0.6, label=' Prediction of Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Perfect Prediction Line')
plt.xlabel('Actual Score ')
plt.ylabel('What the computer predicted')
plt.legend()
plt.show()
next_task(8)

# --- TASK 9: Classification Problem ---
print("\n>>> Task 9: Logistic Regression (Classification)...")
df['Is_High_Score'] = (df['First_Innings_Score'] > 170).astype(int)
y_class = df['Is_High_Score']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_multi, y_class, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(Xc_train, yc_train)
yc_pred = log_reg.predict(Xc_test)
print(f"9.4 Accuracy: {accuracy_score(yc_test, yc_pred)*100:.2f}%")

cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14, "weight": "bold"})
plt.title('9.5 Confusion Matrix: How many correct answers did the model give ?', fontsize=14)
plt.xlabel('What the model said (predicted)', fontsize=12)
plt.ylabel('What actually happened (actually)', fontsize=12)
plt.show()
next_task(9)

# --- TASK 10: Final Metrics & Interpretation ---
print("\n>>> Task 10: Final Metrics...")
mse_val = mean_squared_error(y_test, y_pred_multi)
mae_val = mean_absolute_error(y_test, y_pred_multi)
r2_val = r2_score(y_test, y_pred_multi)

print(f"10.1.1 MSE: {mse_val:.2f}")
print(f"10.1.2 MAE: {mae_val:.2f}")
print(f"10.1.3 R² Score: {r2_val:.2f}")

print("\n10.2 Interpretation:")
print("Regression gives exact runs, while classification gives categories.")
print("10.4: Model performance is excellent for predicting IPL scores.")

print("\n All tasks completed...")