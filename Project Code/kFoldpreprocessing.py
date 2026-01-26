
'''
This follows the behavior from preprocessing.py but for k fold cross validation
The preprocessing will be done for each fold.
The functions will return the folds.

'''


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split


def getPP_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def loadDataset(csvPath, targetColumn="phishing", type="mendeley"):
    #replace -1 with NaN and return separated x and y
    df = pd.read_csv(csvPath)

    X = df.drop(columns=[targetColumn])
    Y = df[targetColumn]

    if type=="uci":
        # Convert labels: -1/1 → 0/1
        Y = (Y == 1).astype(int)

        # Scale features: -1/0/1 → 0/0.5/1
        X = (X + 1) / 2

        # To numpy + reshape for CNN/GRU
        X = X.values.astype("float32")
        #X = X.reshape(X.shape[0], X.shape[1], 1)

    else:
        X = X.replace(-1, np.nan)
        # ---------- Convert to numeric ----------
        X = X.apply(pd.to_numeric, errors="coerce")

    return X, Y


def split_preprocess(csvPath, type="mendeley", targetColumn = "phishing", test_size=0.3, random_state=42):

    X, Y = loadDataset(csvPath, targetColumn, type)

    # ---------- Train-only median imputation ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=test_size,
        stratify=Y,
        random_state=random_state
    )


    if type == "uci":
        # reshape AFTER split
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return X_train, X_val, y_train.values, y_val.values, None
    


    pipeline = getPP_pipeline()
    X_train = pipeline.fit_transform(X_train)
    X_val   = pipeline.transform(X_val)

    return X_train, X_val, y_train.values, y_val.values, pipeline





def kfold_preprocess(csvPath, nSplits = 5, targetColumn = "phishing"):
    
    #This is a generator that will yields preprocessed train/validation folds

    X, Y = loadDataset(csvPath, targetColumn)

    stratKF = StratifiedKFold(
        n_splits = nSplits,
        shuffle = True,
        random_state = 42
    )

    for i, (train_i, test_i) in enumerate(stratKF.split(X, Y), start=1):

        print(f"\n=== Fold {i}/{nSplits} ===")

        X_train_raw = X.iloc[train_i]
        X_val_raw = X.iloc[test_i]

        Y_train = Y.iloc[train_i].values
        Y_val = Y.iloc[test_i].values

        #this will apply the pipeline for training data only
        pipeline = getPP_pipeline()
        X_train = pipeline.fit_transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)

        yield {
            "fold": i,
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "pipeline": pipeline,
            "feature_names": X.columns.tolist()
        }