import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()
