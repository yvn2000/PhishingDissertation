import numpy as np
import tensorflow as tf

import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

import os

os.makedirs("figures", exist_ok=True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, GRU, 
    Dense, Dropout, BatchNormalization, SpatialDropout1D,
    Attention, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from kFoldpreprocessing import kfold_preprocess, split_preprocess

from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


csv_path = "./Datasets/uci.csv"
#epochs = 30
batch_size = 64
test_size=0.2
random_state=42
type="uci"
targetColumn = "Result"


# ---------- Load & split ----------
X_train, X_val, y_train, y_val, pipeline = split_preprocess(csv_path, 
                                                                type=type,
                                                                targetColumn=targetColumn,
                                                                test_size=test_size,
                                                                random_state=random_state
                                                                )



# Add channel dimension ONLY if missing
if X_train.ndim == 2:
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]



inputs = Input(shape=(X_train.shape[1], 1))
model = inputs


model = Conv1D(
            filters=192,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4))(model)

model = BatchNormalization()(model)
model = MaxPooling1D(pool_size=3)(model)

model = Conv1D(
            filters=192,
            kernel_size=7,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4))(model)

model = BatchNormalization()(model)


model = MaxPooling1D(pool_size=2)(model)   # ADD THIS




model = Bidirectional(GRU(
                        256,
                        return_sequences=False, 
                        dropout=0.05,#0.16171145043316665,
                        recurrent_dropout=0.05#0.23988540184445314
                    ))(model)


if len(model.shape) == 3:
    model = GlobalAveragePooling1D()(model)


model = Dense(384, #256,
            activation="relu", 
            kernel_regularizer=l2(1e-4))(model)
model = BatchNormalization()(model)
model = Dropout(0.2552228830393977)(model)


model = Dense(1, activation="sigmoid")(model)
model = Model(inputs, model)



lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=500,
            t_mul=2.0,
            m_mul=0.5,
            alpha=1e-6
        )
    
optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,#lr_schedule,
            weight_decay=1e-4
        )


model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                AUC(name="auc"),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.BinaryAccuracy(
                    name='accuracy', 
                    threshold=0.5 #0.5
                )
            ]
        )

epochs = 15#30
patience = 10 #5

history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_auc",#"val_accuracy",
                        patience=patience,
                        mode="max",
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.3,
                        patience=5,
                        min_lr=1e-6
                    )
                ],
                verbose=2#1
            )


# ---------- Final metrics ----------
y_val_prob = model.predict(X_val).ravel()
y_val_pred = (y_val_prob >= 0.5).astype(int)

cm = confusion_matrix(y_val, y_val_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("figures/confusion.png", dpi=300, bbox_inches='tight')
#plt.show()


'''
fpr, tpr, _ = roc_curve(y_val, y_val_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("figures/roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
'''



roc_auc = roc_auc_score(y_val, y_val_prob)

plt.figure()
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("figures/roc_auc_curve.png", dpi=300, bbox_inches='tight')
#plt.show()




precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_val_prob)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("figures/precision_recall.png", dpi=300, bbox_inches='tight')
#plt.show()



pr_auc = average_precision_score(y_val, y_val_prob)

plt.figure()
plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("figures/avg_precision_recall.png", dpi=300, bbox_inches='tight')
#plt.show()





plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.title("Training vs Validation Loss")
plt.savefig("figures/train_val_loss.png", dpi=300, bbox_inches='tight')
#plt.show()





plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.title("Training vs Validation Accuracy")
plt.savefig("figures/train_val_acc.png", dpi=300, bbox_inches='tight')
#plt.show()






final_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "f1": f1_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_prob)
    }

print(final_metrics)



















