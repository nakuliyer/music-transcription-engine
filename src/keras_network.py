"""
"""

from keras.layers import Dense, Dropout, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, add
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import keras.backend as K

import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import process_audio
from utilities import disp_spec
from dataset import input_windows
from config import *

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def loss(logits, labels):
    """
    Calculates the loss from the logits and the labels
    Args:
        logits: Logits from inference(), float - [batch_size, num_classes]
        labels: Labels tensor, int32 - [batch_size, num_classes]
    Returns:
        cross_entropy: Loss tensor of type float
    """
    print("Logits Shape {}".format(logits.shape))
    print("Labels Shape {}".format(labels.shape))
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits+1e-10)+(1-labels)*tf.log(1-logits+1e-10))
    # factor in musicality
    return cross_entropy

def standard_model():
    input_shape = (window_size, num_freqs)
    inputs = Input(shape=input_shape)
    input_shape_channels = (window_size, num_freqs, 1)
    reshape = Reshape(input_shape_channels)(inputs)

    conv1 = Conv2D(50,(5,25),activation='relu')(reshape)
    #do1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,3))(conv1)

    conv2 = Conv2D(50,(3,5),activation='relu')(pool1)
    #do2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(1,3))(conv2)

    flattened = Flatten()(pool2)
    fc1 = Dense(1160, activation='relu')(flattened)
    do3 = Dropout(0.5)(fc1)

    #fc2 = Dense(200, activation='sigmoid')(do3)
    #do4 = Dropout(0.5)(fc2)
    outputs = Dense(note_range, activation='sigmoid')(do3)
    return inputs, outputs

def resnet_model():
    input_shape = (window_size, num_freqs)
    inputs = Input(shape=input_shape)
    input_shape_channels = (window_size, num_freqs, 1)
    reshape = Reshape(input_shape_channels)(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv = Conv2D(64,(1,int(bins_per_octave*note_range)),padding="same",activation='relu')(reshape)
    pool = MaxPooling2D(pool_size=(1,2))(conv)

    for i in range(int(np.log2(bins_per_octave))-1):
        #print i
        #residual block
        bn = BatchNormalization()(pool)
        re = Activation('relu')(bn)
        freq_range = (bins_per_octave/(2**(i+1)))*note_range
        #print freq_range
        conv = Conv2D(64,(1,int(freq_range)),padding="same",activation='relu')(re)

        #add and downsample
        ad = add([pool,conv])
        pool = MaxPooling2D(pool_size=(1,2))(ad)

    flattened = Flatten()(pool)
    fc = Dense(1024, activation='relu')(flattened)
    do = Dropout(0.5)(fc)
    fc = Dense(512, activation='relu')(do)
    do = Dropout(0.5)(fc)
    outputs = Dense(note_range, activation='sigmoid')(do)
    return inputs, outputs
    #model = Model(inputs=inputs, outputs=outputs)

    #return model

def old_model():
    input_shape = (window_size, num_freqs)
    inputs = Input(shape=input_shape)
    input_shape_channels = (window_size, num_freqs, 1)
    reshape = Reshape(input_shape_channels)(inputs)

    #conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
    conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
    do1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,3))(do1)

    conv2 = Conv2D(50,(3,5),activation='tanh')(pool1)
    do2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(1,3))(do2)

    flattened = Flatten()(pool2)
    fc1 = Dense(1000, activation='sigmoid')(flattened)
    do3 = Dropout(0.5)(fc1)

    fc2 = Dense(200, activation='sigmoid')(do3)
    do4 = Dropout(0.5)(fc2)
    outputs = Dense(note_range, activation='sigmoid')(do4)
    return inputs, outputs

def optimize(type):
    if type == "adam":
        return Adam(lr=alpha, beta_1=beta_1, beta_2=beta_2, decay=decay)
    elif type == "sgd":
        return SGD(lr=alpha, momentum=momentum, decay=decay)
    raise SystemError("Optimizer Type Unrecognized")


class Net:
    def __init__(self, verbose=False, model="std", optimizer="sgd", loss=loss):
        """
        """
        t = time.time()
        # First assert all the types are valid
        self.model_name = "resnet_model"#"cnnmodel--{}--{}--{}".format(model, optimizer, loss)
        model_loc = os.path.join("..", "stored_models", "{}.h5".format(self.model_name))
        if os.path.isfile(model_loc) and False:
            if verbose:
                print("Found Model at {}".format(model_loc))
            self.model = load_model(model_loc)
        else:
            if verbose:
                print("No Model found. Building new Model")
            inputs, outputs = resnet_model()
            self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimize(optimizer),
                           loss=loss,
                           metrics=["acc", f1])
        model_checkpoint = ModelCheckpoint(model_loc, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(patience=5, monitor="val_loss", verbose=1, mode='min')
        csv_logger_name = "resnet_model--{}--{}--{}.log".format(model, optimizer, loss)
        csv_logger = CSVLogger(os.path.join("..", "stored_model_data", csv_logger_name))
        self.callbacks = [model_checkpoint, early_stopping, csv_logger]
        # TODO: Add TensorBoard for verbose
        if verbose:
            self.model.summary()
            print("Initialized Neural Network in {} seconds".format(time.time() - t))

    def train(self, train_gen, val_gen, verbose=False, super_verbose=False, make_plots=True):
        """
        Trains on Model
        """
        verbosity = 0
        if verbose:
            verbosity = 1
        elif super_verbose:
            verbosity = 2
        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=len(train_gen),
                                           epochs=epochs,
                                           verbose=verbosity,
                                           callbacks=self.callbacks,
                                           validation_data=val_gen,
                                           validation_steps=len(val_gen))

        ex_test_file = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_SAMPLE/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M57_AkPnBcht"
        cqt = process_audio.init_cqt(bins_per_octave, sample_rate, thresh, verbose=verbose)

        inputs = input_windows(ex_test_file, cqt)

        #S = []

        #for window in input:
        #    S.append(self.model.predict(np.array([window]), batch_size=batch_size, verbose=1)[0])
        S = self.model.predict(inputs, batch_size=batch_size, verbose=1)

        #S = np.array(S)
        print(S.shape)
        rate, data = process_audio.wav_root_to_data(ex_test_file)
        disp_spec(S.transpose(), len(data), sample_rate, time_step)

        if make_plots:
            # Plot acceleration
            print("History Keys: {}".format(history.history.keys()))
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(os.path.join("..", "stored_model_data", "{}--acc.png".format(self.model_name)))

            # Plot f1
            plt.plot(history.history['f1'])
            plt.plot(history.history['val_f1'])
            plt.title('Model F1 Score')
            plt.ylabel('f1 score')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(os.path.join("..", "stored_model_data", "{}--f1.png".format(self.model_name)))

            # Plot loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(os.path.join("..", "stored_model_data", "{}--los.png".format(self.model_name)))

    def evaluate(self, test_gen, verbose=False):
        evl = self.model.evaluate_generator(test_gen, steps=len(test_gen))
        if verbose:
            print("Evaluated Generator with score of {}".format(evl))
        return evl
