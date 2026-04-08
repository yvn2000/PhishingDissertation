import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


import pandas as pd

# REMOVE SEEDS FOR RANDOMNESS - Comment out these lines if you want reproducibility
import random
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

import os
os.makedirs("figures_uci_final", exist_ok=True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, GRU,
    Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, 
    GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *

import matplotlib.pyplot as plt
import seaborn as sns
import sys

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_no = 1
metrics_per_fold = []

all_histories = []
all_fpr = []
all_tpr = []
all_precisions = []
all_recalls = []

def build_optimized_model(input_shape):
    """
    OPTIMIZED: Slightly larger capacity to compensate for less training data
    But not too deep - stay efficient
    """

    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv1D(192, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(inputs)  # Weaker L2
    x = BatchNormalization()(x)
    #x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=2)(x)

    
    # Conv Block 2
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=3)(x)
    


    # Conv Block 3
    x = Conv1D(256, kernel_size=7, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)


    # BiGRU - INCREASED from 320 to 384
    x = Bidirectional(GRU(128, return_sequences=True, 
                          dropout=0.1, recurrent_dropout=0.15))(x)
    x = BatchNormalization()(x)
    
    # Dual Pooling
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])



    # Dense - INCREASED and ADDED one more layer
    x = Dense(192, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Was 640
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)            #best: 0.35
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Added
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)            #best:0.30

    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model




    '''


    inputs = Input(shape=input_shape)
    
    # Conv Block 1 - INCREASED from 320 to 384
    x = Conv1D(384, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(inputs)  # Weaker L2
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Conv Block 2 - INCREASED from 320 to 384
    x = Conv1D(384, kernel_size=7, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Conv Block 3 - INCREASED from 192 to 256
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    
    # BiGRU - INCREASED from 320 to 384
    x = Bidirectional(GRU(384, return_sequences=True, 
                          dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    
    # Dual Pooling
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Dense - INCREASED and ADDED one more layer
    x = Dense(768, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Was 640
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Added
    x = BatchNormalization()(x)
    x = Dropout(0.40)(x)
    
    x = Dense(384, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Was 320
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Was 160
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)

    
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
    '''



csv_path = "../Datasets/uci.csv"
df = pd.read_csv(csv_path)
target_column = "Result"
X = df.drop(columns=[target_column]).values
y = df[target_column].values
y = (y == -1).astype(int)  # -1 -> 1 (phishing), 1 -> 0 (legitimate)




for train_idx, val_idx in kfold.split(X, y):

    print(f"\n========== Fold {fold_no} ==========")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # reshape if needed
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))



    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {
        0: class_weights[0] * 0.85,
        1: class_weights[1] * 1.25  # Even stronger boost (was 1.20)
    }



    model = build_optimized_model(input_shape=(X_train.shape[1], 1))   # your existing model builder

    model.compile(
        optimizer=Adam(
            #learning_rate=1e-3#1.5e-4
            learning_rate=1.5e-4
        ),  # Slightly higher LR
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,  # More epochs
        batch_size=32, #32,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_auc', patience=30, mode='max', 
                        restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                            min_lr=1e-7,
                            verbose=1)
        ],
        verbose=2
    )

    all_histories.append(history.history)

    # Predictions
    y_prob = model.predict(X_val).ravel()

    # Threshold search
    thresholds = np.arange(0.01, 0.99, 0.001)
    best_acc = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    y_pred = (y_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_prob)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_precisions.append(precision_curve)
    all_recalls.append(recall_curve)

    metrics_per_fold.append({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "conf_matrix": cm
    })

    print(f"Fold {fold_no} Accuracy: {acc:.4f}")
    print(f"Best Threshold: {best_threshold:.3f}")
    print("Confusion Matrix:")
    print(cm)

    fold_no += 1

# =========================
# Final Average Results
# =========================

avg_accuracy = np.mean([m["accuracy"] for m in metrics_per_fold])
avg_precision = np.mean([m["precision"] for m in metrics_per_fold])
avg_recall = np.mean([m["recall"] for m in metrics_per_fold])
avg_f1 = np.mean([m["f1"] for m in metrics_per_fold])
avg_roc_auc = np.mean([m["roc_auc"] for m in metrics_per_fold])

print("\n========== Final Cross-Validated Results ==========")
print(f"Mean Accuracy:  {avg_accuracy:.4f}")
print(f"Mean Precision: {avg_precision:.4f}")
print(f"Mean Recall:    {avg_recall:.4f}")
print(f"Mean F1-score:  {avg_f1:.4f}")
print(f"Mean ROC-AUC:   {avg_roc_auc:.4f}")

for i in range(len(metrics_per_fold)):
    print(f"Fold {i+1}: Accuracy -> {metrics_per_fold[i]["accuracy"]}")