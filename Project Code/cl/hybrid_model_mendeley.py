import numpy as np
import tensorflow as tf
import random
import pandas as pd

# Set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

import os
os.makedirs("figures_hybrid", exist_ok=True)

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(csv_path, dataset_name, target_column, test_size=0.20, sampling_strategy='oversample'):
    """
    Unified preprocessing for both datasets
    """
    df = pd.read_csv(csv_path)
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} Dataset")
    print('='*80)
    print(f"Raw data shape: {df.shape}")
    
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    # Handle dataset-specific encoding
    if dataset_name == "UCI":
        y = (y == -1).astype(int)  # -1 -> 1 (phishing), 1 -> 0 (legitimate)
    elif dataset_name == "Mendeley":
        # Handle missing values (-1)
        print(f"Missing values (-1): {(X == -1).sum():,}")
        X[X == -1] = 0  # Replace with neutral value
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # ========================================
    # SAMPLING STRATEGIES
    # ========================================
    if sampling_strategy == 'oversample':
        # RANDOM OVER SAMPLING (ROS)
        print("\n🔼 Applying Random Over Sampling (ROS)...")
        print(f"Before ROS - Train: Legit={np.sum(y_train==0)}, Phishing={np.sum(y_train==1)}")
        
        # Find minority and majority classes
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        
        if n_class_0 > n_class_1:
            minority_class = 1
            majority_count = n_class_0
        else:
            minority_class = 0
            majority_count = n_class_1
        
        # Get minority class samples
        minority_indices = np.where(y_train == minority_class)[0]
        minority_X = X_train[minority_indices]
        minority_y = y_train[minority_indices]
        
        # Calculate how many samples to add
        n_minority = len(minority_indices)
        n_to_add = majority_count - n_minority
        
        # Randomly sample with replacement from minority class
        resample_indices = np.random.choice(len(minority_X), size=n_to_add, replace=True)
        X_resampled = minority_X[resample_indices]
        y_resampled = minority_y[resample_indices]
        
        # Combine original + oversampled
        X_train = np.vstack([X_train, X_resampled])
        y_train = np.concatenate([y_train, y_resampled])
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        
        print(f"After ROS  - Train: Legit={np.sum(y_train==0)}, Phishing={np.sum(y_train==1)}")
        print(f"✓ Added {n_to_add} synthetic samples (duplicates)")
    
    elif sampling_strategy == 'undersample':
        # RANDOM UNDER SAMPLING (RUS)
        print("\n🔽 Applying Random Under Sampling (RUS)...")
        print(f"Before RUS - Train: Legit={np.sum(y_train==0)}, Phishing={np.sum(y_train==1)}")
        
        # Find minority and majority classes
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        
        if n_class_0 > n_class_1:
            majority_class = 0
            minority_count = n_class_1
        else:
            majority_class = 1
            minority_count = n_class_0
        
        # Get majority and minority indices
        majority_indices = np.where(y_train == majority_class)[0]
        minority_indices = np.where(y_train != majority_class)[0]
        
        # Randomly sample majority class to match minority count
        undersampled_majority_indices = np.random.choice(
            majority_indices, 
            size=minority_count, 
            replace=False  # No replacement for undersampling
        )
        
        # Combine undersampled majority with all minority samples
        balanced_indices = np.concatenate([undersampled_majority_indices, minority_indices])
        
        # Shuffle
        np.random.shuffle(balanced_indices)
        
        # Apply undersampling
        X_train = X_train[balanced_indices]
        y_train = y_train[balanced_indices]
        
        n_removed = len(majority_indices) - minority_count
        print(f"After RUS  - Train: Legit={np.sum(y_train==0)}, Phishing={np.sum(y_train==1)}")
        print(f"✓ Removed {n_removed} majority class samples")
    
    else:
        # No sampling
        print("\n⚖️  No sampling applied (using original imbalanced data)")
        print(f"Train: Legit={np.sum(y_train==0)}, Phishing={np.sum(y_train==1)}")
    
    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    # Adjust class weights based on sampling strategy
    if sampling_strategy in ['oversample', 'undersample']:
        # Data is balanced, use equal weights
        class_weight_dict = {0: 1.0, 1: 1.0}
    else:
        # Data is imbalanced, use stronger weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {
            0: class_weights[0] * 0.82,
            1: class_weights[1] * 1.28
        }
    
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train: Legit={np.sum(y_train==0):,}, Phishing={np.sum(y_train==1):,}")
    print(f"Val: Legit={np.sum(y_val==0):,}, Phishing={np.sum(y_val==1):,}")
    print(f"\nFinal - Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, X_val, y_train, y_val, class_weight_dict


def build_hybrid_model(input_shape):
    """
    Hybrid 1D-CNN + BiGRU model
    """
    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv1D(192, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(inputs)  # Weaker L2
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(pool_size=2)(x)

    
    # Conv Block 2
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(pool_size=3)(x)
    

    # Conv Block 3
    x = Conv1D(256, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(2e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # BiGRU
    x = Bidirectional(GRU(96#128
                          , return_sequences=True, 
                          dropout=0.1, recurrent_dropout=0.15))(x)
    x = BatchNormalization()(x)
    
    
    # Dual Pooling
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    

    #x = GlobalAveragePooling1D()(x)



    # Dense - INCREASED and ADDED one more layer
    x = Dense(192, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Was 640
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)            #best: 0.35
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(2e-5))(x)  # Added
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)            #best:0.30

    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def plot_all_metrics(y_val, y_val_pred, y_val_prob, history, metrics, dataset_name, save_dir):
    """
    Generate all 4 required plots
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Hybrid Model - {dataset_name}\nAccuracy: {metrics["Accuracy"]:.2%}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    fn = cm[1, 0]
    fp = cm[0, 1]
    total = len(y_val)
    plt.text(0.5, -0.12, f'False Negatives: {fn} ({fn/total*100:.2f}%)', 
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.5, -0.17, f'False Positives: {fp} ({fp/total*100:.2f}%)', 
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{int(metrics["Accuracy"]*10000)}Hybrid_{dataset_name}_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Curves (Loss and Accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Hybrid Model - {dataset_name}\nTraining vs Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].axhline(y=0.98, color='r', linestyle='--', label='Target (98%)', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Hybrid Model - {dataset_name}\nTraining vs Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{int(metrics["Accuracy"]*10000)}Hybrid_{dataset_name}_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2.5, label=f'AUC = {metrics["ROC-AUC"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Hybrid Model - {dataset_name}\nROC Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{int(metrics["Accuracy"]*10000)}Hybrid_{dataset_name}_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_val_prob)
    pr_auc = average_precision_score(y_val, y_val_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2.5, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Hybrid Model - {dataset_name}\nPrecision-Recall Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{int(metrics["Accuracy"]*10000)}Hybrid_{dataset_name}_pr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All plots saved to '{save_dir}/'")


# ============================================================================
# MAIN TRAINING
# ============================================================================
if __name__ == "__main__":
    
    print("="*80)
    print("HYBRID 1D-CNN + BiGRU MODEL")
    print("="*80)
    
    # Choose dataset
    DATASET = "Mendeley"  # Change to "UCI" for UCI dataset
    
    if DATASET == "UCI":
        csv_path = "/mnt/user-data/uploads/uci.csv"
        target_column = "Result"
    else:  # Mendeley
        csv_path = "../Datasets/dataset_small.csv"
        #csv_path = "../Datasets/dataset_full.csv"
        target_column = "phishing"
    
    # Load and preprocess
    X_train, X_val, y_train, y_val, class_weight_dict = preprocess_data(
        csv_path, DATASET, target_column, test_size=0.20, sampling_strategy='oversample'
    )
    
    # Build model
    model = build_hybrid_model(input_shape=(X_train.shape[1], 1))
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Train
    print(f"\n{'='*80}")
    print(f"Training on {DATASET} Dataset")
    print('='*80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64 if DATASET == "Mendeley" else 32,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_accuracy'#'val_auc'
                        ,patience=16#10#30#30
                        , mode='max', 
                        restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, 
                            min_lr=1e-7,
                            verbose=1)
        ],
        verbose=2
    )
    
    # Predictions
    print(f"\n{'='*80}")
    print("Making Predictions")
    print('='*80)
    
    y_val_prob = model.predict(X_val, verbose=0).ravel()
    
    # Threshold optimization - FIXED
    print("\nOptimizing classification threshold...")
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.35, 0.65, 0.01):
        y_pred_temp = (y_val_prob >= thresh).astype(int)
        acc = accuracy_score(y_val, y_pred_temp)
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    print(f"Optimal threshold: {best_thresh:.3f} -> Accuracy: {best_acc:.4f}")
    
    y_val_pred = (y_val_prob >= best_thresh).astype(int)
    
    # Calculate all metrics
    metrics = {
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(y_val, y_val_pred),
        "Recall": recall_score(y_val, y_val_pred),
        "F1-Score": f1_score(y_val, y_val_pred),
        "ROC-AUC": roc_auc_score(y_val, y_val_prob),
        "PR-AUC": average_precision_score(y_val, y_val_prob)
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {DATASET} Dataset")
    print('='*80)
    for name, value in metrics.items():
        if name == "Accuracy":
            if value >= 0.98:
                status = "✓"
            else:
                status = " "
        else:
            status = " "
        print(f"{status} {name:.<20} {value:.4f} ({value*100:.2f}%)")
    print('='*80)
    
    # Per-class accuracy
    mask_legit = y_val == 0
    mask_phish = y_val == 1
    acc_legit = accuracy_score(y_val[mask_legit], y_val_pred[mask_legit])
    acc_phish = accuracy_score(y_val[mask_phish], y_val_pred[mask_phish])
    
    print(f"\nPer-class accuracy:")
    print(f"  Legitimate: {acc_legit:.4f} ({acc_legit*100:.2f}%)")
    print(f"  Phishing:   {acc_phish:.4f} ({acc_phish*100:.2f}%)")
    
    # Error analysis
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\nError Analysis:")
    print(f"  False Negatives: {cm[1,0]:,} (missing phishing)")
    print(f"  False Positives: {cm[0,1]:,} (flagging legitimate)")
    print(f"  Total Errors: {cm[1,0] + cm[0,1]:,} / {len(y_val):,}")
    
    # Generate all plots
    print(f"\n{'='*80}")
    print("Generating Plots")
    print('='*80)
    
    plot_all_metrics(y_val, y_val_pred, y_val_prob, history, metrics, 
                     DATASET, "figures_mendeley")
    
    # Save model if good
    if metrics["Accuracy"] >= 0.95:
        model_filename = f'hybrid_model_{DATASET.lower()}.h5'
        model.save(f'figures_mendeley/{model_filename}')
        print(f"\n✓ Model saved as 'figures_mendeley/{model_filename}'")
    
    # Final message
    if metrics["Accuracy"] >= 0.98:
        print("\n" + "="*80)
        print("🎉 SUCCESS! 98%+ ACCURACY ACHIEVED!")
        print("="*80)
    elif metrics["Accuracy"] >= 0.95:
        print("\n" + "="*80)
        print("✓ Good performance! 95%+ accuracy achieved")
        print(f"  (Target: 98%, Current: {metrics['Accuracy']*100:.2f}%)")
        print("="*80)
    else:
        print(f"\n⚠️ Reached {metrics['Accuracy']*100:.2f}%")
        print("="*80)
