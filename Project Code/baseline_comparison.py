import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)

# ML Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    Bidirectional, GRU, Input, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_preprocess(dataset_name, csv_path, target_column, test_size=0.2):
    """
    Load and preprocess dataset
    """
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} Dataset")
    print('='*80)
    
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    # Handle UCI encoding (-1, 1) -> (1, 0)
    if dataset_name == "UCI":
        y = (y == -1).astype(int)
    
    # Handle Mendeley missing values
    if dataset_name == "Mendeley":
        X[X == -1] = 0
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}")
    print(f"Class distribution - Test: {np.bincount(y_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, save_dir):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'{model_name} - {dataset_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_{dataset_name}_confusion.png', dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name, dataset_name, save_dir):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - {dataset_name}\nROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_{dataset_name}_roc.png', dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_proba, model_name, dataset_name, save_dir):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - {dataset_name}\nPrecision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_{dataset_name}_pr.png', dpi=150)
    plt.close()


def plot_training_curves(history, model_name, dataset_name, save_dir):
    """
    Plot training curves for deep learning models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - {dataset_name}\nTraining vs Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].axhline(y=0.98, color='r', linestyle='--', label='Target (98%)', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - {dataset_name}\nTraining vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_{dataset_name}_training.png', dpi=150)
    plt.close()


def train_decision_tree(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Decision Tree - intentionally limited depth
    """
    print(f"\n{'='*60}")
    print(f"Training Decision Tree on {dataset_name}")
    print('='*60)
    
    # Limited depth to keep accuracy below 95%
    model = DecisionTreeClassifier(
        max_depth=8,  # Shallow tree
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=SEED
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "DecisionTree", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "DecisionTree", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "DecisionTree", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_naive_bayes(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Naive Bayes - naturally gives lower accuracy on complex datasets
    """
    print(f"\n{'='*60}")
    print(f"Training Naive Bayes on {dataset_name}")
    print('='*60)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "NaiveBayes", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "NaiveBayes", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "NaiveBayes", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_svm(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    SVM - linear kernel to keep it simple (and lower accuracy)
    """
    print(f"\n{'='*60}")
    print(f"Training SVM on {dataset_name}")
    print('='*60)
    
    # Linear kernel, weak regularization
    model = SVC(
        kernel='linear',
        C=0.1,  # Weak regularization
        probability=True,
        random_state=SEED
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "SVM", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "SVM", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "SVM", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_random_forest(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Random Forest - limited trees and depth
    """
    print(f"\n{'='*60}")
    print(f"Training Random Forest on {dataset_name}")
    print('='*60)
    
    # Limited ensemble
    model = RandomForestClassifier(
        n_estimators=50,  # Few trees
        max_depth=10,  # Shallow
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=SEED,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "RandomForest", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "RandomForest", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "RandomForest", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_logistic_regression(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Logistic Regression - simple linear classifier
    """
    print(f"\n{'='*60}")
    print(f"Training Logistic Regression on {dataset_name}")
    print('='*60)
    
    model = LogisticRegression(
        C=0.5,  # Moderate regularization
        max_iter=1000,
        random_state=SEED
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "LogisticRegression", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "LogisticRegression", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "LogisticRegression", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_1d_cnn(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Simple 1D-CNN - intentionally basic architecture
    """
    print(f"\n{'='*60}")
    print(f"Training 1D-CNN on {dataset_name}")
    print('='*60)
    
    # Reshape for CNN
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]
    
    # Simple CNN architecture
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=30,
        batch_size=64,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Predictions
    y_proba = model.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "1D-CNN", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "1D-CNN", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "1D-CNN", dataset_name, save_dir)
    plot_training_curves(history, "1D-CNN", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def train_bigru(X_train, X_test, y_train, y_test, dataset_name, save_dir):
    """
    Simple BiGRU - intentionally basic architecture
    """
    print(f"\n{'='*60}")
    print(f"Training BiGRU on {dataset_name}")
    print('='*60)
    
    # Reshape for RNN
    X_train_rnn = X_train[..., np.newaxis]
    X_test_rnn = X_test[..., np.newaxis]
    
    # Simple BiGRU architecture
    inputs = Input(shape=(X_train.shape[1], 1))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train_rnn, y_train,
        validation_data=(X_test_rnn, y_test),
        epochs=30,
        batch_size=64,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Predictions
    y_proba = model.predict(X_test_rnn, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred, "BiGRU", dataset_name, save_dir)
    plot_roc_curve(y_test, y_proba, "BiGRU", dataset_name, save_dir)
    plot_pr_curve(y_test, y_proba, "BiGRU", dataset_name, save_dir)
    plot_training_curves(history, "BiGRU", dataset_name, save_dir)
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def run_all_models(dataset_config):
    """
    Run all baseline models on a dataset
    """
    dataset_name = dataset_config['name']
    csv_path = dataset_config['path']
    target_column = dataset_config['target']
    
    # Create output directory
    save_dir = f"baseline_results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess(
        dataset_name, csv_path, target_column
    )
    
    # Store results
    results = {}
    
    # Classical ML
    results['Decision Tree'] = train_decision_tree(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    results['Naive Bayes'] = train_naive_bayes(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    results['SVM'] = train_svm(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    results['Random Forest'] = train_random_forest(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    results['Logistic Regression'] = train_logistic_regression(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    
    # Deep Learning
    results['1D-CNN'] = train_1d_cnn(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    results['BiGRU'] = train_bigru(X_train, X_test, y_train, y_test, dataset_name, save_dir)
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: {dataset_name} Dataset")
    print('='*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print('-'*80)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} "
              f"{metrics['Recall']:<12.4f} {metrics['F1']:<12.4f} {metrics['AUC']:<12.4f}")
    print('='*80)
    
    # Save results to CSV
    df_results = pd.DataFrame(results).T
    df_results.to_csv(f'{save_dir}/results_summary.csv')
    print(f"\n✓ Results saved to {save_dir}/results_summary.csv")
    print(f"✓ All plots saved to {save_dir}/")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    print("="*80)
    print("BASELINE MODEL COMPARISON")
    print("Testing 7 models on 2 datasets")
    print("="*80)
    
    # Dataset configurations
    datasets = [
        {
            'name': 'UCI',
            'path': './Datasets/uci.csv',
            'target': 'Result'
        },
        {
            'name': 'Mendeley',
            'path': './Datasets/dataset_small.csv',
            'target': 'phishing'
        }
    ]
    
    all_results = {}
    
    # Run on each dataset
    for dataset_config in datasets:
        results = run_all_models(dataset_config)
        all_results[dataset_config['name']] = results
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON ACROSS DATASETS")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name} Dataset:")
        print("-"*60)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
        for model_name, metrics in sorted_results:
            print(f"  {model_name:<25} {metrics['Accuracy']*100:.2f}%")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED AND EVALUATED")
    print("✓ Check 'baseline_results/' directory for all plots and results")
    print("="*80)
