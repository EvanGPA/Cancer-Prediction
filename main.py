import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("breast-cancer.csv")
data = data.drop(columns=["Unnamed: 32", "id"])

diagnosis = []

for value in data.diagnosis:
    if value == "M":
        diagnosis.append(1)
    else:
        diagnosis.append(0)

data["diagnosis"] = diagnosis

y = data["diagnosis"]
x = data.drop(["diagnosis"], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 42)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)

print(classification_report(Y_test, y_pred))

data["diagnosis"].value_counts().plot(kind = "bar")

plt.xlabel("Diagnosis")
plt.ylabel("Number of patients")
plt.title("Breast Cancer Diagnosis Distribution")
plt.show()