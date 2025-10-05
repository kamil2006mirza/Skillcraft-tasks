# Decision Tree Classifier with Bank Marketing Dataset

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

# Load dataset
data = pd.read_csv("bank.csv")
data.head()

# Encode categorical data
labelencoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = labelencoder.fit_transform(data[col])

# Split data into train and test
X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Predict on test data
y_pred = dtree.predict(X_test)

# Evaluate
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plot_confusion_matrix(dtree, X_test, y_test)
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(dtree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Fit Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Plot confusion matrix
plot_confusion_matrix(rf, X_test, y_test)
plt.show()
