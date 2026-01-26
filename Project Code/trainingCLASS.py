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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from kFoldpreprocessing import kfold_preprocess








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
                self.model = MaxPooling1D(pool_size=layersConv[i][5])(self.model)
                layerJSON['max_pooling_size'] = layersConv[i][5]
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

def doRun(epochs, batch_size, n_splits, dataset_path, conv_layers, bigru_layers, dense_layers):

    '''
    How setup looks like:
    {
        epochs: 5,
        batch_size: 32,
        n_splits: 3,
        dataset_path: ......,
        conv_layers: [
            [
                num of filters,
                kernel size,
                activation function,
                padding,
                kernel regularizer,
                Max Pooling pool size (0 means no max pooling)      MPooling happens after batchnormalization
            ],
            ....
        ],
        bigru_layers: [
            [
                GRU units,
                return sequences value -> True for stacking RNNs, attention, dual pooling (batch_size, timesteps, features), False is (batch_size, features)
                dropout percentage (0.3 means drop 30% of input)
                recurrent dropout (0.2 means drop 20% of recurrent state connections)
            ],
            ....
        ],
        dense_layers: [
            [
                num of units,
                activation function,
                kernel regularizer,
                dropout
            ],
            ...
        ]
    }
    '''

    EPOCHS = epochs
    BATCH_SIZE = batch_size #32#64
    N_SPLITS = n_splits

    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=10,
        restore_best_weights=True,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='max'
    )

    metrics_per_fold = []

    finalJSON = {}
    finalJSON['epochs'] = EPOCHS
    finalJSON['batch_size'] = BATCH_SIZE
    finalJSON['n_splits'] = N_SPLITS
    #csv="./Datasets/dataset_full.csv"
    #csv = "./Datasets/dataset_small.csv"
    csv = dataset_path
    finalJSON['dataset'] = csv

    flag = 0

    for fold_data in kfold_preprocess(csvPath=csv, nSplits=N_SPLITS):

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



        modelX = Train((X_train.shape[1], 1))
        modelX.addGaussianNoise(0.01)   #noise value

        '''
        ConvLayer = [
            num of filters,
            kernel size,
            activation function,
            padding,
            kernel regularizer,
            Max Pooling pool size (0 means no max pooling)      MPooling happens after batchnormalization
        ]
        Each followed by: BatchNormalization
        '''
        layersConv = [
            [128, 5, "relu", "same", l2(1e-4), 2],
            [256, 3, "relu", "same", l2(1e-4), 2],
            [512, 3, "relu", "same", l2(1e-4), 0]
        ]

        layersConv = conv_layers


        '''
        BiGRULayer = [
            GRU units,
            return sequences value -> True for stacking RNNs, attention, dual pooling (batch_size, timesteps, features), False is (batch_size, features)
            dropout percentage (0.3 means drop 30% of input),
            recurrent dropout (0.2 means drop 20% of recurrent state connections)
        ]
        '''
        layersBiGRU = [
            [128, True, 0.3, 0.2, True],         #last is attention
            [64, True, 0.3, 0.2, False]
        ]

        layersBiGRU = bigru_layers

        modelX.addLayers(layersConv, layersBiGRU, dualPooling=True)

        '''
        DenseLayer = [
            num of units,
            activation function,
            kernel regularizer,
            dropout
        ]
        '''
        layersDense = [
            [256, "relu", l2(1e-4), 0.5],
            [128, "relu", l2(1e-4), 0.4]
        ]

        layersDense = dense_layers
        
        modelX.addDenseLayers(layersDense)
        modelX.addOutputLayer([1, "sigmoid"])

        modelX.compileModel()

        #print(modelX.trainJSON)

        if (flag==0):
            flag = 1
            finalJSON = finalJSON | modelX.trainJSON


        #add model checkpointing. this was recommended, idk what it does
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"best_model_fold_{fold}.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=0
        )

        finalModel = modelX.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            class_weight=class_weight,          #new
            verbose=1
        )


        # predictions
        y_val_prob = modelX.model.predict(X_val).ravel()
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

        finalJSON[f"fold_{fold}"] = fold_metrics


        #print(f"Fold {fold} best threshold: {best_thresh:.3f}")
        print(f"Fold {fold} metrics:", fold_metrics)
        metrics_per_fold.append(fold_metrics)

        break #remove later
    
    print("\n===== Cross-Validation Results =====")
    res = {}
    for metric in metrics_per_fold[0].keys():
        values = [m[metric] for m in metrics_per_fold]
        res[metric] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        print(
            f"{metric}: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )
    
    #print(res)
    finalJSON['cv_results'] = res
    #print(finalJSON)
    #print(finalJSON['cv_results']['accuracy'])
    #accuracy_str = next(iter(finalJSON['cv_results']['accuracy']))
    accuracy_str = finalJSON['cv_results']['accuracy']
    #print(accuracy_str)
    acc = float(accuracy_str.split('±')[0].strip())
    #print(acc)
    acc = int(str(acc).split('.')[1].strip())
    #print(acc)
    write_dict_to_json(finalJSON, "tests_non_optuna", acc)

    print("Run Done")





if __name__ == '__main__':

    '''
    ConvLayer = [
        num of filters,
        kernel size,
        activation function,
        padding,
        kernel regularizer,
        Max Pooling pool size (0 means no max pooling)      MPooling happens after batchnormalization
    ]
    Each followed by: BatchNormalization

    BiGRULayer = [
        GRU units,
        return sequences value -> True for stacking RNNs, attention, dual pooling (batch_size, timesteps, features), False is (batch_size, features)
        dropout percentage (0.3 means drop 30% of input),
        recurrent dropout (0.2 means drop 20% of recurrent state connections)
    ]

    DenseLayer = [
        num of units,
        activation function,
        kernel regularizer,
        dropout
    ]
    '''


    '''
    doRun(
        epochs = 1,
        batch_size = 64,
        n_splits = 2,
        dataset_path = "./Datasets/dataset_small.csv",
        conv_layers = [
            [128, 5, "relu", "same", l2(1e-4), 2],
            [256, 3, "relu", "same", l2(1e-4), 0]
        ],
        bigru_layers = [
            [64, True, 0.3, 0.2, True]         #last is attention
        ],
        dense_layers = [
            [128, "relu", l2(1e-4), 0.4]
        ]
    )


    doRun(
        epochs = 2,
        batch_size = 32,
        n_splits = 3,
        dataset_path = "./Datasets/dataset_small.csv",
        conv_layers = [
            [128, 5, "relu", "same", l2(1e-4), 2],
            [256, 3, "relu", "same", l2(1e-4), 2],
            [512, 3, "relu", "same", l2(1e-4), 0]
        ],
        bigru_layers = [
            [128, True, 0.3, 0.2, True],         #last is attention
            [64, True, 0.3, 0.2, False]
        ],
        dense_layers = [
            [256, "relu", l2(1e-4), 0.5],
            [128, "relu", l2(1e-4), 0.4]
        ]
    )

    '''

    doRun(
        epochs = 20,
        batch_size = 64,
        n_splits = 4,
        dataset_path = "./Datasets/dataset_small.csv",
        conv_layers = [
            [128, 5, "relu", "same", l2(1e-4), 2],
            [256, 3, "relu", "same", l2(1e-4), 2]
        ],
        bigru_layers = [
        ],
        dense_layers = [
            [128, "relu", l2(1e-4), 0.4]
        ]
    )


    doRun(
        epochs = 20,
        batch_size = 64,
        n_splits = 4,
        dataset_path = "./Datasets/dataset_small.csv",
        conv_layers = [
        ],
        bigru_layers = [
            [64, True, 0.3, 0.2, False]
        ],
        dense_layers = [
            [128, "relu", l2(1e-4), 0.4]
        ]
    )
    
    








