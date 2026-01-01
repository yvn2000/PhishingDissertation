'''
Process so i don't forget.

Clean the data
Handle invalid/missing/unknown values (instead of -1 for empty, use NaN or null)
Normalize features correctly (z-score)
Save the preprocessed dataset separately for fast reuse
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os


print("================= PRE-PROCESSING =================")


mendeley2020_full_PATH = "./Datasets/dataset_full.csv"
processed_mendeley_2020_full_PATH = "./Datasets/mendeley_2020_full_preprocessed.csv"
mendeley_2020_SCALER_PATH = "./Datasets/mendeley_2020_full_feature_scaler.joblib"

df = pd.read_csv(mendeley2020_full_PATH)
print("Dataset shape:", df.shape)
#print(f"Features: {df.columns.tolist()}")
print(f"Num of features: {len(df.columns.tolist())} (should be 112, including the class/target attribute)")

print("================= X and Y Split =================")
target = "phishing"

print(f"Phishing: {(df[target] == 1).sum()}")
print(f"Legitimate: {(df[target] == 0).sum()}")

X = df.drop(columns=[target])
Y = df[target]
print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("...Replacing -1 with NaNs")
X = X.replace(-1, np.nan)


print("================= PIPELINE =================")

#pipeline is used to prevent data leakage
preP_pipeline = Pipeline([
    #imputer is for replacing NaNs with medians
    ("imputer", SimpleImputer(strategy="median")),      #safest compromise as ANNs nead to have meaningful numbers
    #scaler is used because features have varying scales of values:
    #qty_dot_url is from 0-10 while ttl_hostname is from 0-86400, etc
    ("scaler", StandardScaler())
    #the scaler normalizes the values using z scrore which uses mean and standard deviation
    #i.e. x = (x - mean) / std
])

'''
We can't just remove rows with -1 or unknown values as these are actual
websites that may be missed if we do so. Doing so adds in a sort of
bias. We don't want URLs that are easy to analyze.
'''


#apply the pipeline
X_cleaned = preP_pipeline.fit_transform(X)

X_processed_df = pd.DataFrame(
    X_cleaned,
    columns=X.columns
)
processed_df = pd.concat([X_processed_df, Y.reset_index(drop=True)], axis=1)

processed_df.to_csv(processed_mendeley_2020_full_PATH, index=False)
joblib.dump(preP_pipeline, mendeley_2020_SCALER_PATH)

print("Saved:", processed_mendeley_2020_full_PATH)
print("Saved scaler:", mendeley_2020_SCALER_PATH)


#print(processed_df.describe())
print(processed_df.isnull().sum().sum(), "missing values remaining")
print(processed_df[target].value_counts())

print("================= PREPROCESSED SAVED =================")










