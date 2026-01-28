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


import optuna





class Train:

    

    def __init__(self, input_shape):

        self.gaussianNoise = 0
        self.inputs = Input(shape=input_shape)
        self.model = self.inputs
        self.trainJSON = {}
        

    def addGaussianNoise(self,value=0.01):
        self.model = tf.keras.layers.GaussianNoise(value)(self.model)
        self.trainJSON['gaussian_noise'] = value
        #print(f"tf.keras.layers.GaussianNoise({value})(self.model)")
        #print()
        #print()

    
    def addLayers(self, layersConv, layersBiGRU, dualPooling):
        '''
        layersConv =[
            [128, 5, "relu", "same", l2(1e-4), 2],
            [256, 3, "relu", "same", l2(1e-4), 2]
        ]
        '''
        convList = []
        for i in range(len(layersConv)):
            layerJSON = {}
            self.model = Conv1D(
                filters=layersConv[i][0],
                kernel_size=layersConv[i][1],
                activation=layersConv[i][2],
                padding=layersConv[i][3],
                kernel_regularizer=layersConv[i][4])(self.model)
            
            layerJSON['n_filters'] = layersConv[i][0]
            layerJSON['kernel_size'] = layersConv[i][1]
            layerJSON['activation_function'] = layersConv[i][2]
            layerJSON['padding'] = True
            layerJSON['kernel_regularizer'] = 'l2(1e-4)'

            #print(f"Conv1D(filters={layersConv[i][0]}, kernel_size={layersConv[i][1]}, activation={layersConv[i][2]}, padding={layersConv[i][3]}, kernel_regularizer=l2(1e-4))(self.model)")

            self.model = BatchNormalization()(self.model)

            layerJSON['batch_normalization'] = True

            #print(f"BatchNormalization()(self.model)")

            if ((i==(len(layersConv)-1)) and (len(layersBiGRU)>0)):
                layerJSON['max_pooling_size'] = 0
                convList.append(layerJSON)
                #print()
                break
            else:
                current_length = self.model.shape[1]  # sequence length
                pool_size = min(layersConv[i][5], current_length)
                self.model = MaxPooling1D(pool_size=pool_size)(self.model)
                layerJSON['max_pooling_size'] = pool_size

                #self.model = MaxPooling1D(pool_size=layersConv[i][5])(self.model)
                #layerJSON['max_pooling_size'] = layersConv[i][5]

                #print(f"MaxPooling1D(pool_size={layersConv[i][5]})(self.model)")
            
            #print(layerJSON)
            convList.append(layerJSON)
            #print()

        #print()
        #print(convList)
        self.trainJSON['convolutional_layers'] = convList
        '''
        layersBiGRU =[
            [128, True, 0.3, 0.2, True],         #last is attention
            [64, True, 0.3, 0.2, False]
        ]
        '''
        
        bigruList = []
        for i in range(len(layersBiGRU)):
            layerJSON = {}
            self.model = Bidirectional(GRU(
                                        layersBiGRU[i][0],
                                        return_sequences=layersBiGRU[i][1], 
                                        dropout=layersBiGRU[i][2],
                                        recurrent_dropout=layersBiGRU[i][3]
                                    ))(self.model)
            layerJSON['n_gru_units'] = layersBiGRU[i][0]
            layerJSON['return_sequences'] = layersBiGRU[i][1]
            layerJSON['dropout'] = layersBiGRU[i][2]
            layerJSON['recurrent_dropout'] = layersBiGRU[i][3]
            #print(f"Bidirectional(GRU({layersBiGRU[i][0]}, return_sequences={layersBiGRU[i][1]}, dropout={layersBiGRU[i][2]}, recurrent_dropout={layersBiGRU[i][3]}))(self.model)")
            #print()

            if ((layersBiGRU[i][1] is True) and (layersBiGRU[i][4] is True)):

                attention = Attention()([self.model, self.model])
                #print(f"Attention()([self.model, self.model])")

                self.model = Concatenate()([self.model, attention])
                #print(f"Concatenate()([self.model, attention])")
                layerJSON['attention'] = True
                #print()
            else:
                layerJSON['attention'] = False
    

            if ((dualPooling is True) and (i == len(layersBiGRU)-1) and (layersBiGRU[i][1] is True)):

                avg_pool = GlobalAveragePooling1D()(self.model)
                #print(f"GlobalAveragePooling1D()(self.model)")

                max_pool = GlobalMaxPooling1D()(self.model)
                #print(f"GlobalMaxPooling1D()(self.model)")
                
                self.model = Concatenate()([avg_pool, max_pool])
                #print(f"Concatenate()([avg_pool, max_pool])")
                layerJSON['dual_pooling'] = True
                #print()
            else:
                layerJSON['dual_pooling'] = False
            
            bigruList.append(layerJSON)
            #print()
        
        #print(bigruList)
        self.trainJSON['bigru_layers'] = bigruList

    
    def addDenseLayers(self, layersDense):
        if len(self.model.shape) == 3:
            self.model = GlobalAveragePooling1D()(self.model)
            self.trainJSON['beforeDense'] = 'global_average_pooling'
            #print(f"self.model = GlobalAveragePooling1D()(self.model)")
            #self.model = Flatten()(self.model)
            #self.trainJSON['beforeDense'] = 'flatten'

        '''
        [
            [256, "relu", l2(1e-4), 0.5],
            [128, "relu", l2(1e-4), 0.4]
        ]
        '''
        denseList = []
        for i in range(len(layersDense)):
            layerJSON = {}
            self.model = Dense(layersDense[i][0], activation=layersDense[i][1], 
                    kernel_regularizer=layersDense[i][2])(self.model)
            
            layerJSON['n_dense'] = layersDense[i][0]
            layerJSON['activation_function'] = layersDense[i][1]
            layerJSON['kernel_regularizer'] = 'l2(1e-4)'
            #print(f"dense{i+1} = Dense({layersDense[i][0]}, activation={layersDense[i][1]}, kernel_regularizer=l2(1e-4))(dense{i+1})")
            
            self.model = BatchNormalization()(self.model)
            layerJSON['batch_normalization'] = True
            #print(f"dense{i+1} = BatchNormalization()(dense{i+1})")

            self.model = Dropout(layersDense[i][3])(self.model)
            layerJSON['dropout'] = layersDense[i][3]
            #print(f"dense{i+1} = Dropout({layersDense[i][3]})(dense{i+1})")
            #print()

            denseList.append(layerJSON)
        #print(denseList)
        self.trainJSON['dense_layers'] = denseList


    def addOutputLayer(self, outputLayer):
        '''
        [1, "sigmoid]
        '''
        self.model = Dense(outputLayer[0], activation=outputLayer[1])(self.model)
        #print(f"self.model = Dense({outputLayer[0]}, activation={outputLayer[1]})(self.model)")
        self.trainJSON['output_layer'] = {'n_dense': outputLayer[0], 'function': outputLayer[1]}

        self.model = Model(self.inputs, self.model)
        #print(f"self.model = Model(self.inputs, self.model)")
        #print()

    
    def compileModel(self):
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

        self.model.compile(
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



import json
import os
from datetime import datetime

#CHATGPT CODE
def write_dict_to_json(data_dict, directory, acc):
    """
    Writes a Python dictionary to a JSON file in the specified directory.
    The filename is automatically generated with a timestamp for uniqueness.

    :param data_dict: Dictionary to write
    :param directory: Directory where the JSON file will be saved
    """
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #acc = round(acc, 4) * 1000
        filename = f"test_{int(acc)}_{timestamp}.json"
        file_path = os.path.join(directory, filename)

        # Write the dictionary to the JSON file
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

        print(f"Dictionary successfully written to {file_path}")
        return file_path

    except Exception as e:
        print(f"Error writing dictionary to JSON: {e}")
        return None







import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def optuna_Best(
    run_name,
    csv_path,
    type,
    epochs,
    batch_size,
    n_trials=30,
    test_size=0.2,
    random_state=42,
    save_dir="tests"
):
    targetColumn = ""
    if type=="mendeley":
        targetColumn = "phishing"
    elif type=="uci":
        targetColumn = "Result"
    
    # ---------- Load & split ----------
    X_train, X_val, y_train, y_val, pipeline = split_preprocess(csv_path, 
                                                                type=type,
                                                                targetColumn=targetColumn,
                                                                test_size=test_size,
                                                                random_state=random_state
                                                                )

    '''
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    '''

    print("Preprocessed")

    '''
    # CNN input shape
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    '''

    # Add channel dimension ONLY if missing
    if X_train.ndim == 2:
        X_train = X_train[..., np.newaxis]
        X_val   = X_val[..., np.newaxis]

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    RUN_NAME = run_name

    # ---------- Optuna objective ----------
    def on_trial_complete(study, trial):
        print(f"Trial {trial.number} completed — value: {trial.value:.4f}")

    def objective(trial):

        try:
            print()
            print(f"Trial {trial.number} started")

            modelX = Train((X_train.shape[1], 1))
            
            #modelX.addGaussianNoise(0.01)          #no noise for trials, add in for final resulting model

            # ----- CNN -----
            nConv = trial.suggest_int("nConv", 1, 3)
            layersConv = []

            for i in range(nConv):
                pool_size = trial.suggest_int(f"pool_{i}", 2, min(4, X_train.shape[1] // 2))
                layersConv.append([
                    trial.suggest_int(f"filters_{i}", 64, 256, step=64),
                    trial.suggest_int(f"kernel_{i}", 3, 7, step=2),
                    "relu",
                    "same",
                    l2(1e-4),
                    pool_size
                ])

            print("CNN suggested")

            # ----- BiGRU -----
            n_gru = trial.suggest_int("n_gru", 0, 2)
            layersBiGRU = []

            for i in range(n_gru):
                layersBiGRU.append([
                    trial.suggest_int(f"gru_units_{i}", 64, 256, step=64),
                    i < n_gru - 1,
                    trial.suggest_float(f"gru_dropout_{i}", 0.1, 0.4),
                    trial.suggest_float(f"gru_rec_dropout_{i}", 0.1, 0.3),
                    False
                ])

            print("BiGRU suggested")

            modelX.addLayers(layersConv, layersBiGRU, dualPooling=True)

            # ----- Dense -----
            n_dense = trial.suggest_int("n_dense", 1, 3)
            layersDense = []

            for i in range(n_dense):
                layersDense.append([
                    trial.suggest_int(f"dense_units_{i}", 64, 256, step=64),
                    "relu",
                    l2(1e-4),
                    trial.suggest_float(f"dense_dropout_{i}", 0.2, 0.6)
                ])

            print("Dense Layers suggested")

            modelX.addDenseLayers(layersDense)
            modelX.addOutputLayer([1, "sigmoid"])
            modelX.compileModel()

            print("Suggested model compiled")

            print("Started Training")
            print(f"Conv Layers: {nConv}, BiGRU Layers: {n_gru}, Dense Layers: {n_dense}")

            history = modelX.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",#"val_auc",
                        patience=5,
                        mode="max",
                        restore_best_weights=True
                    )
                ],
                verbose=2#1
            )

            print("Suggested model trained")
            print(f"Trial {trial.number} done")

            #return max(history.history["val_auc"])
            best_val_acc = max(history.history["val_accuracy"])
        
            trial_json = {
                "trial_number": trial.number,
                "params": trial.params,
                "best_val_accuracy": best_val_acc,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "epochs_ran": len(history.history["loss"]),
                "model_structure": modelX.trainJSON
            }

            write_dict_to_json(
                trial_json,
                directory=f"tests/optuna_trials/{RUN_NAME}",
                acc=int(best_val_acc * 100)
            )

            return best_val_acc
        
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            write_dict_to_json(
                {"trial_number": trial.number, "error": str(e)},
                directory=f"tests/optuna_trials/{RUN_NAME}/failures",
                acc=0
            )
            raise optuna.exceptions.TrialPruned()

    print("Optuna study started")
    # ---------- Run Optuna ----------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[on_trial_complete]
    )

    best_params = study.best_params

    # ---------- Retrain BEST model ----------
    print("Best model optained")
    modelX = Train((X_train.shape[1], 1))
    modelX.addGaussianNoise(0.01)

    # rebuild CNN
    layersConv = []
    for i in range(best_params["nConv"]):
        layersConv.append([
            best_params[f"filters_{i}"],
            best_params[f"kernel_{i}"],
            "relu",
            "same",
            l2(1e-4),
            best_params[f"pool_{i}"]
        ])

    # rebuild BiGRU
    layersBiGRU = []
    if best_params["n_gru"] > 0:
        for i in range(best_params["n_gru"]):
            layersBiGRU.append([
                best_params[f"gru_units_{i}"],
                i < best_params["n_gru"] - 1,
                best_params[f"gru_dropout_{i}"],
                best_params[f"gru_rec_dropout_{i}"],
                False
            ])

    modelX.addLayers(layersConv, layersBiGRU, dualPooling=True)

    # rebuild Dense
    layersDense = []
    for i in range(best_params["n_dense"]):
        layersDense.append([
            best_params[f"dense_units_{i}"],
            "relu",
            l2(1e-4),
            best_params[f"dense_dropout_{i}"]
        ])

    modelX.addDenseLayers(layersDense)
    modelX.addOutputLayer([1, "sigmoid"])
    print("Best model recreated")
    modelX.compileModel()
    print("Best model recompiled")

    modelX.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=10,
                mode="max",
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    print("Best model retrained")

    # ---------- Final metrics ----------
    y_val_prob = modelX.model.predict(X_val).ravel()
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    final_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "f1": f1_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_prob)
    }

    final_json = {
        "dataset": csv_path,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_params": best_params,
        "val_metrics": final_metrics
    }

    write_dict_to_json(final_json, save_dir, int(final_metrics["accuracy"] * 100))
    print("Best model json saved")
    return final_json








if __name__ == '__main__':
    
    csv = "./Datasets/uci.csv"
    type="uci"
    n_trials=30
    epochs=20
    batch_size=64
    test_size=0.3

    jsonRes = optuna_Best(
        run_name=f"uci_{epochs}epochs_{batch_size}batch_{n_trials}trials",
        csv_path = csv,
        type=type,
        epochs=epochs,
        batch_size=batch_size,
        n_trials=n_trials,
        test_size=test_size
    )






    csv = "./Datasets/dataset_small.csv"
    type="mendeley"
    n_trials=30
    epochs=20
    batch_size=64
    test_size=0.3

    jsonRes = optuna_Best(
        run_name=f"mendeleySmall_{epochs}epochs_{batch_size}batch_{n_trials}trials",
        csv_path = csv,
        type=type,
        epochs=epochs,
        batch_size=batch_size,
        n_trials=n_trials,
        test_size=test_size
    )







    csv = "./Datasets/uci.csv"
    type="uci"
    n_trials=30
    epochs=20
    batch_size=128
    test_size=0.3

    jsonRes = optuna_Best(
        run_name=f"uci_{epochs}epochs_{batch_size}batch_{n_trials}trials",
        csv_path = csv,
        type=type,
        epochs=epochs,
        batch_size=batch_size,
        n_trials=n_trials,
        test_size=test_size
    )







    csv = "./Datasets/dataset_small.csv"
    type="mendeley"
    n_trials=30
    epochs=20
    batch_size=128
    test_size=0.3

    jsonRes = optuna_Best(
        run_name=f"mendeleySmall_{epochs}epochs_{batch_size}batch_{n_trials}trials",
        csv_path = csv,
        type=type,
        epochs=epochs,
        batch_size=batch_size,
        n_trials=n_trials,
        test_size=test_size
    )






    




    
    








