
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


def getPP_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def loadDataset(csvPath, targetColumn="phishing"):
    #replace -1 with NaN and return separated x and y
    df = pd.read_csv(csvPath)

    X = df.drop(columns=[targetColumn])
    Y = df[targetColumn]

    X = X.replace(-1, np.nan)

    return X, Y


def kfold_preprocess(csvPath, nSplits = 5, targetColumn = "phishing"):
    
    #This is a generator that will yields preprocessed train/validation folds

    X, Y = loadDataset(csvPath, targetColumn)

    skf = StratifiedKFold(
        n_splits = nSplits,
        shuffle = True,
        random_state = 42
    )

    for i, (train_i, test_i) in enumerate(skf.split(X, Y), start=1):

        print(f"\n=== Fold {i}/{nSplits} ===")

        X_train_raw = X.iloc[train_i]
        X_val_raw = X.iloc[test_i]

        Y_train = Y.iloc[train_i].values
        Y_val = Y.iloc[test_i].values

        #this will create a pipeline for training data only
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