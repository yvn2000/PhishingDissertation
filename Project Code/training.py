'''

OKay so here will be the CNN and BiGRU

CNN will take in data of shape (num_samples, num_features)
which is like (~80000, 111)

CNN + BiGRU, it will be (num_samples, num_features, 1)
num_features = sequence length
1 = channel dimension

--- Functionalities ---
1D-CNN:
- local/spatial features
- reduce noise before recurrent layers?

MaxPooling:
- reduce dimensionality

BiGRU:
- learn temporal dependencies
- learn contextual relationships

Dense Layers:
- final binary classification


For binary classification:
Output layer activation function will be sigmoid
and loss function will be binary cross entropy

For preventing overfitting, maybe use earlystopping and reducelr. Dont worry for now.

Across folds, results will be aggregated using mean and standard deviation.

NoTE: FOR NOW THE MODELS ARE NOT BEING SAVED, AND ALSO HYPERPARAM TUNING WILL COME LATER

dont forget metric cillection


'''

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, GRU, 
    Dense, Dropout, BatchNormalization, SpatialDropout1D,
    Attention, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

from tensorflow.keras.regularizers import l2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from kFoldpreprocessing import kfold_preprocess


#print(f"TensorFlow Version: {tf.__version__}")
#print("Import successful!")

'''
Threshold optimizing ended up making things worse,
wont be used, atleast for now.
'''
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_acc = 0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh, best_acc


def build_cnn_bigru(input_shape):
    """
    Conv -> BN -> Pool
    Conv -> BN
    BiGRU
    GlobalPooling
    Dense
    """


    inputs = Input(shape=input_shape)

    #add noise for robustness (regularization)
    x = tf.keras.layers.GaussianNoise(0.01)(inputs)



    x = Conv1D(filters=128, kernel_size=5, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)



    x = Conv1D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)



    x = Conv1D(filters=512, kernel_size=3, activation="relu", 
               padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    #x = Bidirectional(GRU(units=64, return_sequences=False, dropout=0.2))(x)

    #x = Bidirectional(GRU(units=64, return_sequences=False, dropout=0.2))(x)

    #x = GlobalMaxPooling1D()(x)

    # BiGRU layers with skip connections
    gru1 = Bidirectional(GRU(128, return_sequences=True, 
                           dropout=0.3, recurrent_dropout=0.2))(x)
    
    # Attention mechanism
    attention = Attention()([gru1, gru1])
    
    # Combine GRU output with attention
    gru_concat = Concatenate()([gru1, attention])
    
    gru2 = Bidirectional(GRU(64, return_sequences=True, 
                           dropout=0.3, recurrent_dropout=0.2))(gru_concat)
    
    # Dual pooling strategy
    avg_pool = GlobalAveragePooling1D()(gru2)
    max_pool = GlobalMaxPooling1D()(gru2)
    concat_pool = Concatenate()([avg_pool, max_pool])



    #x = Dense(64, activation="relu")(x)
    #x = Dropout(0.5)(x)

    dense1 = Dense(256, activation="relu", 
                   kernel_regularizer=l2(1e-4))(concat_pool)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation="relu", 
                   kernel_regularizer=l2(1e-4))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)

    #outputs = Dense(1, activation="sigmoid")(x)
    outputs = Dense(1, activation="sigmoid")(dense2)

    model = Model(inputs, outputs)

    '''
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[AUC(name="auc")]
    )
    '''

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=500,
        t_mul=2.0,
        m_mul=0.5,
        alpha=1e-6
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
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
                threshold=0.5
            )
        ]
    )

    return model



if __name__ == '__main__':

    EPOCHS = 50
    BATCH_SIZE = 32#64
    N_SPLITS = 5

    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=10,
        restore_best_weights=True,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_lauc",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='max'
    )

    metrics_per_fold = []

    for fold_data in kfold_preprocess(
        #csvPath="./Datasets/dataset_full.csv",
        csvPath="./Datasets/dataset_small.csv",
        nSplits=N_SPLITS
    ):

        fold = fold_data["fold"]
        X_train = fold_data["X_train"]
        y_train = fold_data["Y_train"]
        X_val = fold_data["X_val"]
        y_val = fold_data["Y_val"]


        neg, pos = np.bincount(y_train)
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)  #standard weighting
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        print(f"\n===== Training Fold {fold} =====")

        #reshape for CNN input
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]


        model = build_cnn_bigru(
            input_shape=(X_train.shape[1], 1)
        )

        #add model checkpointing. this was recommended, idk what it does
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"best_model_fold_{fold}.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=0
        )

        finalModel = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight,          #new
            verbose=1
        )

        # predictions
        y_val_prob = model.predict(X_val).ravel()
        #best_thresh, best_acc = find_best_threshold(y_val, y_val_prob)  #threshold optimization
        y_val_pred = (y_val_prob >= 0.4).astype(int)
        #y_val_pred = (y_val_prob >= 0.5).astype(int)
        #y_val_pred = (y_val_prob >= best_thresh).astype(int)

        fold_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred),
            "roc_auc": roc_auc_score(y_val, y_val_prob),
            #"best_threshold": best_thresh
        }

        #print(f"Fold {fold} best threshold: {best_thresh:.3f}")
        print(f"Fold {fold} metrics:", fold_metrics)
        metrics_per_fold.append(fold_metrics)

        #break #remove later


    print("\n===== Cross-Validation Results =====")

    for metric in metrics_per_fold[0].keys():
        values = [m[metric] for m in metrics_per_fold]
        print(
            f"{metric}: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )


