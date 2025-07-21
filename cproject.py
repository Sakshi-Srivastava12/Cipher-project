import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("diabetic.csv")
print(df)
print("Initial Shape:", df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("\n❓ Missing Values:")
print(df.isnull().sum())
print()
#CLEANING DATA
df['Age'] = df['Age'].fillna(df['Age'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())
df['Diabetic'] = df['Diabetic'].fillna(df['Diabetic'].median())
df['FamilyHistory'] = df['FamilyHistory'].map({'Yes': 1, 'No': 0})
df['Diabetic'] = df['Diabetic'].round().astype(int)
#EDA
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")
sns.countplot(data=df,x='Diabetic', hue='Diabetic', palette=['skyblue', 'orange','pink'],legend=False)
plt.title('Diabetes Count', fontsize=14)
plt.xlabel('Diabetic', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap")
plt.show()
#SPLIT DATA
X = df.drop('Diabetic', axis=1)
y = df['Diabetic']

#Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

#Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)


# Logistic Regression
log_reg = LogisticRegression(class_weight='balanced')  # helps with class imbalance
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

#Conclusion
if accuracy_score(y_test, log_preds) > accuracy_score(y_test, tree_preds):
    print("\n Logistic Regression performed better.")
else:
    print("\n Decision Tree performed better.")
