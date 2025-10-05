import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset (update path if needed)
data = pd.read_csv(r"J:\python_titanic_graph\titanic.csv")

# Clean column names (lowercase for consistency)
data.columns = data.columns.str.lower()

sns.set_style("whitegrid")

# --- Plot 1: Age Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(data['age'].dropna(), bins=40, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# --- Plot 2: Survival Count by Gender ---
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="sex", hue="survived", palette={0: "green", 1: "orange"})
plt.title("Survival Count by Gender")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# --- Plot 3: Survival Count by Passenger Class ---
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="pclass", hue="survived", palette={0: "red", 1: "blue"})
plt.title("Survival Count by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()
