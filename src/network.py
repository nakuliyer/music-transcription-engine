"""
Methods for building a neural network

author: Nakul Iyer
date: 2/28/19
"""

from keras.layers import Dense, Dropout, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, add, Activation
from keras.models import load_model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras.layers.normalization import BatchNormalization
import keras.backend as K

import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import os
import matplotlib.pyplot as plt

import process_audio
from utilities import disp_spec, disp_roll
from dataset import input_windows
from config import *

class TimeHistory(Callback):
    def __init__(self, fname):
        self.times = []
        self.total_time = 0
        self.fname = "{}.txt".format(fname)

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        epoch_time = time.time() - self.epoch_time_start
        self.times.append(epoch_time)
        self.total_time = epoch_time
        with open(self.fname, "w+") as f:
            f.write(str(self.times))
            f.write(str(self.total_time))

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

def baseline_gpu_model():
    input_shape = (window_size, num_freqs)

    # Normally, RGB images have 3 channel, but
    # spectrograms only have one channel
    input_shape_channels = (window_size, num_freqs, 1)

    model = Sequential()
    model.add(Reshape(input_shape_channels, input_shape=input_shape))

    model.add(Conv2D(50, (3, 25)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Conv2D(50, (3, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(88))
    model.add(Activation("sigmoid"))
    return model

def note_cnn():
    input_shape = (window_size, num_freqs)

    # Normally, RGB images have 3 channel, but
    # spectrograms only have one channel
    input_shape_channels = (window_size, num_freqs, 1)
    model = Sequential()
    model.add(Reshape(input_shape_channels, input_shape=input_shape))

    model.add(Conv2D(32, (5, 4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(note_range))
    model.add(Activation("sigmoid"))
    return model

def draw_plots(model_name, history_log, keys=["acc", "f1", "loss"]):
    """
    Draw plots for ACC, LOSS, and F1 and stores in
    ../stored_model_data/
    Parameters
    ----------
    model_name : string
                 Name of the model for storing purpose
    history_log : dict
                  Dictionary with keys in keys
    """
    print(history_log.keys())

    for key in history_log:
        if key not in keys:
            print("Could Not Draw Graphs")
            return

    plt.plot(history_log['acc'])
    if "val_acc" in history_log:
        plt.plot(history_log['val_acc'])
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join("..", "stored_model_data", "{}--acc.png".format(model_name)))
    plt.close()

    # Plot f1
    plt.plot(history_log['f1'])
    if "val_f1" in history_log:
        plt.plot(history_log['val_f1'])
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.title('Model F1 Score')
    plt.ylabel('f1 score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join("..", "stored_model_data", "{}--f1.png".format(model_name)))
    plt.close()

    # Plot loss
    plt.plot(history_log['loss'])
    if "val_loss" in history_log:
        plt.plot(history_log['val_loss'])
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join("..", "stored_model_data", "{}--los.png".format(model_name)))
    plt.close()

def punish_row(row_idx):
    """Prevent model from predicting higher notes too often"""
    return np.abs(note_range / 2 - row_idx) * 2 * punish_factor / note_range + 1
    #return row_idx / note_range + 1

def get_model(model_name):
    if model_name == "std":
        return baseline_gpu_model()
    elif model_name == "std_gpu":
        return baseline_gpu_model()
    elif model_name == "note_cnn":
        return note_cnn()
    raise SystemError("Model Type Unrecognized")

def get_optimizer(optimizer_name):
    if optimizer_name == "adam":
        return Adam(lr=alpha, beta_1=beta_1, beta_2=beta_2, decay=decay)
    elif optimizer_name == "sgd":
        return SGD(lr=alpha, momentum=momentum, decay=decay)
    raise SystemError("Optimizer Type Unrecognized")

def get_verbosity(verbose, super_verbose):
    verbosity = 0
    if verbose:
        verbosity = 1
    elif super_verbose:
        verbosity = 2
    return verbosity

class Net:
    def __init__(self,
                 verbose=False,
                 model="std_gpu",
                 optimizer="adam",
                 loss="binary_crossentropy",
                 train_spe=train_spe,
                 use_cuda=True,
                 reload_model=True,
                 use_partial_memory=0,
                 early_stop_on="val_loss"):
        """Initializes the Neural Network"""
        t = time.time()

        if verbose:
            print("Available GPUs: {}".format(K.tensorflow_backend._get_available_gpus()))

        # GPU setup method calls
        cpu_count = 1
        gpu_count = 1 if use_cuda else 0
        config = tf.ConfigProto(intra_op_parallelism_threads=4,\
                inter_op_parallelism_threads=4, allow_soft_placement=True,\
                device_count = {'CPU' : cpu_count, 'GPU' : gpu_count})
        if use_partial_memory > 0:
            config.gpu_options.per_process_gpu_memory_fraction = use_partial_memory
        sess = tf.Session(config=config)
        K.set_session(sess)

        # Check if model exists in model_ckpts, then load else create new
        self.model_name = "{}_model--train_spe={}--max_frames={}--optimizer={}--loss={}".format(model, train_spe, input_max_frames, optimizer, loss if type(loss) is str else "custom")
        model_loc = os.path.join("..", "model_ckpts", "{}.h5".format(self.model_name))
        if reload_model:
            if os.path.isfile(model_loc):
                if verbose:
                    print("Found Model at {}".format(model_loc))
                self.model = load_model(model_loc, custom_objects={"f1": f1})
            else:
                # We're being asked to reload a file which doesn't exist
                raise SystemError("Model could not be loaded at {}. Try using different parameters to find a model that exists at ../model_ckpts".format(model_loc))
        else:
           if verbose:
               print("No Model found. Building new Model")
           self.model = get_model(model)
        self.model.compile(get_optimizer(optimizer),
                           loss=loss,
                           metrics=["acc", f1])
        self.history = None
        model_checkpoint = ModelCheckpoint(model_loc, monitor="val_loss", verbose=1, save_best_only=False, mode='min')
        early_stopping = EarlyStopping(patience=10, monitor=early_stop_on, verbose=1, mode='min')
        csv_logger_name = "{}_model--train_spe={}--max_frames={}--optimizer={}--loss={}.log".format(model, train_spe, input_max_frames, optimizer, loss if type(loss) is str else "custom")
        csv_logger = CSVLogger(os.path.join("..", "stored_model_data", csv_logger_name))
        time_history = TimeHistory(self.model_name)
        self.callbacks = [model_checkpoint, early_stopping, csv_logger, time_history]
        if verbose:
            self.model.summary()
            print("Initialized Neural Network in {} seconds".format(time.time() - t))

    def train(self, train_gen, val_gen, verbose=False, super_verbose=False, make_plots=True):
        """
        Trains on Model
        """
        verbosity = get_verbosity(verbose, super_verbose)
        if super_verbose:
            print("Train Gen SPE: {} \nVal Gen SPE: {}".format(train_spe, val_spe))
        self.history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=int(train_spe),
                                           epochs=epochs,
                                           verbose=verbosity,
                                           callbacks=self.callbacks,
                                           validation_data=val_gen,
                                           validation_steps=int(val_spe))
        if make_plots:
            self.draw_history_plots()

    def run_test(self, ex_test_file, verbose=False, super_verbose=False, output_avg_of=100):
        """Tests the model with a given test file"""
        verbosity = get_verbosity(verbose, super_verbose)

        if verbose:
            print("Getting Spectrogram for test file: {}".format(ex_test_file))

        cqt = process_audio.init_cqt(bins_per_octave, sample_rate, thresh, verbose=verbose)
        rate, data = process_audio.wav_root_to_data(ex_test_file)
        if not rate == sample_rate:
            # We'll never really fall in here
            # Since scipy.io.wavfile.read always
            # Uses a rate of 44.1 kHz but just in case
            raise ValueError("rate {} does not equal hyper-parameter sample_rate, {}".format(rate, sample_rate))
        specgram = process_audio.constant_q_transform(data, cqt, sample_rate, time_step)
        test_inputs = input_windows(specgram)
        test_num_windows = test_inputs.shape[0]

        # Split the song into smaller windows if it is too large
        test_cases = []
        if test_num_windows >= input_max_frames:
            i = 0
            while i * input_max_frames < test_num_windows:
                input_case = test_inputs[input_max_frames * i: min(input_max_frames * (i+1), len(test_inputs) - 1)]
                test_cases.append(input_case)
                i += 1
        else:
            test_cases.append(test_inputs)

        sum_S = None
        for output_ft_spec in range(output_avg_of):
            S = None
            for test_case in test_cases:
                test_case_prediction = np.array(self.model.predict(test_case, batch_size=1, verbose=verbosity))
                if S is None:
                    S = test_case_prediction
                else:
                    S = np.concatenate((S, test_case_prediction), axis=0)

            # change S from (time, midi_notes) to (midi_notes, time)
            S = np.abs(S.transpose())
            for row_idx in range(S.shape[0]):
                S[row_idx][S[row_idx] < output_mask_threshold * punish_row(row_idx)] = 0
                S[row_idx][S[row_idx] >= output_mask_threshold * punish_row(row_idx)] = 1
            if sum_S is None:
                sum_S = S
            else:
                sum_S = np.maximum(sum_S, S)
        avg_S = sum_S
        if verbose:
            disp_roll(avg_S, len(data), sample_rate, time_step)
        return avg_S

    def draw_history_plots(self):
        draw_plots(self.model_name, self.history.history)

    def evaluate(self, test_gen, verbose=False):
        evl = self.model.evaluate_generator(test_gen, steps=test_spe)
        if verbose:
            print("Evaluated Generator with score of {}".format(evl))
        return evl
