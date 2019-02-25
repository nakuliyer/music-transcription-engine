"""
"""

from keras.layers import Dense, Dropout, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, add, Activation
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import keras.backend as K

import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import os
import matplotlib.pyplot as plt
import process_audio
from utilities import disp_spec
from dataset import input_windows
from config import *

class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = "std2-17.h5"
            self.model.save_weights(name)
        self.batch += 1

class GraphSaver(Callback):
    def __init__(self, model_name):
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs={}):
        draw_plots(self.model_name, logs)

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

def other_model():
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
    outputs = Dense(note_range, activation='relu')(do3)
    return inputs, outputs

def standard_model():
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

def old_baseline_model():
    input_shape = (window_size, num_freqs)

    # Normally, RGB images have 3 channel, but
    # spectrograms only have one channel
    input_shape_channels = (window_size, num_freqs, 1)

    model = Sequential()
    model.add(Reshape(input_shape_channels, input_shape=input_shape))

    model.add(Conv2D(50, (5, 25), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Conv2D(50, (3, 5), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Flatten())
    model.add(Dense(1000, activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(note_range, activation='sigmoid'))
    return model

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

def get_model(model_name):
    if model_name == "std":
        return baseline_model()
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
    def __init__(self, verbose=False, model="std_gpu", optimizer="adam", loss="binary_crossentropy", use_cuda=True, reload_model=True, use_partial_memory=0, early_stop_on="val_loss"):
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
        if reload_model and os.path.isfile(model_loc):
           if verbose:
               print("Found Model at {}".format(model_loc))
           self.model = load_model(model_loc, custom_objects={"f1": f1})
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
        graph_saver = GraphSaver(self.model_name)
        self.callbacks = [model_checkpoint, early_stopping, csv_logger, graph_saver]
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

        avg_S = None
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
            S[S < output_mask_threshold] = 0
            S[S >= output_mask_threshold] = 1
            if avg_S is None:
                avg_S = S
            else:
                avg_S = np.minimum(avg_S, S)
        if verbose:
            disp_spec(avg_S, len(data), sample_rate, time_step, reduce=False)
        return avg_S

    def draw_history_plots(self):
        draw_plots(self.model_name, self.history.history)

    def evaluate(self, test_gen, verbose=False):
        evl = self.model.evaluate_generator(test_gen, steps=test_spe)
        if verbose:
            print("Evaluated Generator with score of {}".format(evl))
        return evl
