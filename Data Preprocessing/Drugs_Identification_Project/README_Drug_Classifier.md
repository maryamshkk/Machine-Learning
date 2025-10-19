# 💊 Decision Tree Classifier – Drug Classification

## 🧠 Project Overview  
This project implements a **Decision Tree Classifier** to predict which **drug type (DrugA, DrugB, DrugC, DrugX, or DrugY)** a patient should be prescribed based on their **Age, Sex, Blood Pressure (BP), Cholesterol**, and **Na_to_K (sodium-to-potassium ratio)** levels.

The model uses the **entropy (information gain)** criterion to split data and determine classification rules.  
The final decision tree visualization clearly shows **how each feature contributes to predicting the correct drug**.

---

## 📂 Dataset
**Dataset Name:** Drugs A, B, C, X, Y  
**Source:** [Kaggle – Drugs Classification Dataset](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y)

### 📊 Attributes
| Feature | Description |
|----------|--------------|
| Age | Age of the patient |
| Sex | Gender (Male/Female) |
| BP | Blood Pressure (Low/Normal/High) |
| Cholesterol | Cholesterol level (Normal/High) |
| Na_to_K | Sodium-to-Potassium ratio |
| Drug | Target label – Type of drug prescribed (A/B/C/X/Y) |

---

## ⚙️ Project Steps

### 🧩 **Step 1 – Import Libraries**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
```

---

### 📥 **Step 2 – Load Dataset**
```python
data = pd.read_csv("drug200.csv")
```
- Viewed shape, columns, and missing values.

---

### 🔍 **Step 3 – Data Understanding**
Checked dataset summary:
```python
data.info()
data.describe()
data['Drug'].value_counts()
```

---

### 🧼 **Step 4 – Data Preprocessing**
Encoded categorical variables:
```python
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['BP'] = le.fit_transform(data['BP'])
data['Cholesterol'] = le.fit_transform(data['Cholesterol'])
```
Split into **features** and **target**:
```python
X = data.drop('Drug', axis=1)
y = data['Drug']
```

---

### ✂️ **Step 5 – Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

### 🌳 **Step 6 – Train Decision Tree Model**
```python
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(X_train, y_train)
```

---

### 📊 **Step 7 – Model Evaluation**
```python
y_pred = dtc.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

✅ **Accuracy:** `1.0 (100%)`  
✅ **Perfect predictions across all drug classes**

---

### 🌲 **Step 8 – Decision Tree Visualization**
```python
plt.figure(figsize=(20,10))
plot_tree(dtc, filled=True, feature_names=X.columns, class_names=y.unique(), rounded=True)
plt.title("Decision Tree Visualization from Drug Classification")
plt.show()
```

📸 The visualization shows how the model splits data:
- Top features like **Na_to_K** and **BP** guide most decisions.
- Each leaf node represents a **predicted drug**.

---

## 📈 Results Summary

| Metric | Score |
|---------|--------|
| Accuracy | 100% |
| Precision | 1.0 |
| Recall | 1.0 |
| F1-score | 1.0 |

✅ The model performs perfectly on the dataset due to clear separability among classes.

---

## 🧩 Key Insights
- **Na_to_K (Sodium-to-Potassium ratio)** is the **most important deciding factor**.  
- Decision Tree uses **entropy** to split data for maximum information gain.  
- The model can be visualized for better interpretability — useful in **healthcare** and **medical decision systems**.

---

## 🧰 Requirements
Install dependencies before running:
```bash
pip install pandas numpy scikit-learn matplotlib graphviz
```

If you face this error:
```
ExecutableNotFound: failed to execute 'dot', make sure the Graphviz executables are on your system's PATH
```
Then install **Graphviz** manually:  
🔗 [https://graphviz.gitlab.io/download/](https://graphviz.gitlab.io/download/)

---

## 💡 Key Learnings
- How to preprocess categorical data.  
- How Decision Trees use entropy and information gain.  
- How to visualize and interpret model decisions.  
- Importance of clean datasets for accurate predictions.

---

## 👩‍💻 Author
**Maryam Sheikh**  
🎓 Codveda Internship – *Intermediate Level Task: Decision Tree Classifier*

---

## 🏁 Conclusion
This project successfully demonstrates how a **Decision Tree Classifier** can be used to predict drug prescriptions.  
It provides clear visualization, high interpretability, and perfect performance on a clean medical dataset.

---

✨ *End of Project Report – Decision Tree Drug Classification* ✨
