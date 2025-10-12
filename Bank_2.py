import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
Data_Frame = pd.read_csv("/Users/bored/OneDrive/Desktop/Jypiter Folder/nlp/bank+marketing/bank/bank.csv", sep = ";")
print(Data_Frame)

# Separate features and target
X = Data_Frame.drop('y', axis=1)
y = Data_Frame['y'].map({'yes': 1, 'no': 0})  # encode target

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original feature count")

