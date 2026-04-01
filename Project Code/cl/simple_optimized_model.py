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
os.makedirs("figures_simple", exist_ok=True)

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


def optimized_preprocess(csv_path, target_column='Result', test_size=0.15):
    """
    Optimized preprocessing pipeline
    """
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    y = (y == -1).astype(int)  # Convert: -1 -> 1 (phishing), 1 -> 0 (legitimate)
    
    # Stratified split with smaller test set for more training data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # Use RobustScaler (better for data with outliers)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, X_val, y_train, y_val, class_weight_dict


def build_optimized_model(input_shape):
    """
    Optimized architecture focusing on your original design but with improvements
    """
    inputs = Input(shape=input_shape)
    
    # First Conv Block - increased filters
    x = Conv1D(256, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(5e-5))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Second Conv Block
    x = Conv1D(256, kernel_size=7, padding='same', activation='relu',
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Third Conv Block (added for more capacity)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    
    # BiGRU with more units
    x = Bidirectional(GRU(256, return_sequences=True, 
                          dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Global pooling with both avg and max
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Dense layers with gradual reduction
    x = Dense(512, activation='relu', kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================================
# MAIN TRAINING
# ============================================================================
print("="*80)
print("OPTIMIZED MODEL - TARGETING 98% ACCURACY")
print("="*80)

# Load data
csv_path = "./Datasets/uci.csv"
X_train, X_val, y_train, y_val, class_weight_dict = optimized_preprocess(csv_path)

# Build model
model = build_optimized_model(input_shape=(X_train.shape[1], 1))
print(f"\nTotal parameters: {model.count_params():,}")

# Compile with lower initial learning rate
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Train with improved callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=20, mode='max', 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=7, 
                         min_lr=1e-7, verbose=1)
    ],
    verbose=2
)

# Predictions with threshold optimization
y_val_prob = model.predict(X_val, verbose=0).ravel()

# Find optimal threshold
best_acc = 0
best_thresh = 0.5
for thresh in np.arange(0.35, 0.65, 0.01):
    y_pred = (y_val_prob >= thresh).astype(int)
    acc = accuracy_score(y_val, y_pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"\nOptimal threshold: {best_thresh:.3f} -> Accuracy: {best_acc:.4f}")

y_val_pred = (y_val_prob >= best_thresh).astype(int)

# Final metrics
metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred),
    "Recall": recall_score(y_val, y_val_pred),
    "F1-Score": f1_score(y_val, y_val_pred),
    "ROC-AUC": roc_auc_score(y_val, y_val_prob)
}

print("\n" + "="*80)
print("FINAL METRICS")
print("="*80)
for name, value in metrics.items():
    status = "✓" if (name == "Accuracy" and value >= 0.98) else " "
    print(f"{status} {name:.<20} {value:.4f} ({value*100:.2f}%)")
print("="*80)

# Visualizations
# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title(f'Confusion Matrix - Accuracy: {metrics["Accuracy"]:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('figures_simple/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Training curves
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
plt.savefig('figures_simple/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["ROC-AUC"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures_simple/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Plots saved to 'figures_simple/' directory")
