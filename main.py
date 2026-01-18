import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("heart.csv")
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())

#Histogram
for label in df.columns[:-1]:
    plt.hist(df[df["target"]==1][label],color='blue',label='Yes',alpha=0.7,density=True)
    plt.hist(df[df["target"]==0][label],color='green',label='No',alpha=0.7,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    # plt.show()

#Class Distribution
plt.figure(figsize=(5,4))
df["target"].value_counts().plot(kind='bar')
plt.xticks([0,1],["No(0)","Yes(1)"],rotation =0 )
plt.ylabel("Count")
plt.title("Class Distribution")
# plt.show()

#Correlation
corr_matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt = ".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()

#Categorical Feature encoding
cat_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
# print(df.head())

#Feature Marging
df['age_chol'] = df['age'] * df['chol']
df['age_thalach'] = df['age'] * df['thalach']
df['chol_thalach'] = df['chol'] * df['thalach']

#split
X = df.drop(columns=['target'])
y = df['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

#Scaling
num_cols = X_train.select_dtypes(include=['int64','float64']).columns

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val_scaled[num_cols]   = scaler.transform(X_val[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

df_scaled = pd.DataFrame(X_train_scaled)
# print(df_scaled.head())

#Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='liblinear'
)
lr.fit(X_train_scaled,y_train)

val_pred_lr = lr.predict(X_val_scaled)
# print("Logistic Regression\n",accuracy_score(y_val,val_pred_lr))

#Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train_scaled,y_train)
val_pred_rf = rf.predict(X_val_scaled)
print("Logistic Regression\n",accuracy_score(y_val,val_pred_rf))

