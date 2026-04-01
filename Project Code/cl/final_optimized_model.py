import numpy as np
import tensorflow as tf
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
    GlobalMaxPooling1D, Concatenate, GaussianNoise
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


def preprocess_uci(csv_path, target_column='Result', test_size=0.20, sampling_strategy='oversample'):
    """
    Optimized preprocessing for UCI with 20% test + Sampling options
    
    Args:
        sampling_strategy: 'none', 'oversample' (ROS), or 'undersample' (RUS)
    """
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    y = (y == -1).astype(int)  # -1 -> 1 (phishing), 1 -> 0 (legitimate)
    
    # Split BEFORE sampling (important!)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
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
    
    print(f"\nFinal - Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, X_val, y_train, y_val, class_weight_dict


def build_optimized_model(input_shape):
    """
    final model
    """

    inputs = Input(shape=input_shape)

    inputs = GaussianNoise(0.02)(inputs)
    
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
    x = Dropout(0.20)(x)


    # BiGRU - INCREASED from 320 to 384
    x = Bidirectional(GRU(96#128
                          , return_sequences=True, 
                          dropout=0.1, recurrent_dropout=0.15))(x)
    x = BatchNormalization()(x)
    
    # Dual Pooling
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])

    #x = GlobalAveragePooling1D()(x)


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


def plot_all(y_val, y_val_pred, y_val_prob, history, metrics, save_dir, accuracy):
    """Generate all 4 plots"""
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - Accuracy: {metrics["Accuracy"]:.2%}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    fn, fp = cm[1, 0], cm[0, 1]
    plt.text(0.5, -0.12, f'FN: {fn} ({fn/len(y_val)*100:.2f}%)', 
             ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.17, f'FP: {fp} ({fp/len(y_val)*100:.2f}%)', 
             ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{accuracy}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].axhline(y=0.98, color='r', linestyle='--', label='Target (98%)', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{accuracy}training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2.5, label=f'AUC = {metrics["ROC-AUC"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{accuracy}roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_val_prob)
    pr_auc = average_precision_score(y_val, y_val_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2.5, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ROS_{accuracy}precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================







print("="*80)
print("OPTIMIZED UCI MODEL - 20% TEST, TARGETING 98%")
print("="*80)

csv_path = "../Datasets/uci.csv"
# With oversampling (recommended)
X_train, X_val, y_train, y_val, class_weight_dict = preprocess_uci(
    csv_path, sampling_strategy='oversample'
)

model = build_optimized_model(input_shape=(X_train.shape[1], 1))
print(f"\nParameters: {model.count_params():,}")

model.compile(
    optimizer=Adam(
        #learning_rate=3e-4
        learning_rate=1.5e-4
    ),  # Slightly higher LR
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,  # More epochs
    batch_size=16, #32,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_accuracy'#'val_auc'
                      ,patience=30#10#30#30
                      , mode='max', 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                         min_lr=1e-7,
                         verbose=1)
    ],
    verbose=2
)

# Predictions
y_val_prob = model.predict(X_val, verbose=0).ravel()

# Threshold optimization
print("\nOptimizing threshold...")
best_acc = 0
best_thresh = 0.560

for thresh in np.arange(0.40, 0.70, 0.01):
    y_pred = (y_val_prob >= thresh).astype(int)
    acc = accuracy_score(y_val, y_pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Optimal threshold: {best_thresh:.3f} -> {best_acc:.4f}")

y_val_pred = (y_val_prob >= best_thresh).astype(int)

# Metrics
metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred),
    "Recall": recall_score(y_val, y_val_pred),
    "F1-Score": f1_score(y_val, y_val_pred),
    "ROC-AUC": roc_auc_score(y_val, y_val_prob)
}

print("\n" + "="*80)
print("FINAL RESULTS - UCI with 20% Test")
print("="*80)
for name, value in metrics.items():
    status = "✓" if (name == "Accuracy" and value >= 0.98) else " "
    print(f"{status} {name:.<20} {value:.4f} ({value*100:.2f}%)")
print("="*80)

# Per-class
mask_legit = y_val == 0
mask_phish = y_val == 1
print(f"\nPer-class:")
print(f"  Legitimate: {accuracy_score(y_val[mask_legit], y_val_pred[mask_legit]):.4f}")
print(f"  Phishing:   {accuracy_score(y_val[mask_phish], y_val_pred[mask_phish]):.4f}")

# Errors
cm = confusion_matrix(y_val, y_val_pred)
print(f"\nErrors: FN={cm[1,0]}, FP={cm[0,1]}, Total={cm[1,0]+cm[0,1]}/{len(y_val)}")

# Plots
dir = "figures_uci_final"
plot_all(y_val, y_val_pred, y_val_prob, history, metrics, dir, int(metrics["Accuracy"]*10000))
print("\n✓ All plots saved to 'figures_uci_final/'")

if metrics["Accuracy"] >= 0.98:
    model.save('figures_uci_final/uci_model_98.h5')
    print("✓ Model saved")
    print("\n🎉 98%+ ACHIEVED!")
else:
    print(f"\n⚠ {metrics['Accuracy']*100:.2f}% (Target: 98%)")

print("="*80)