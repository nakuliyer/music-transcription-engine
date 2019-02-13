"""
Loading and manipulating dataset
The MAPS class is a data generator
TODO:
- Minibatch process rather than stream currently will be faster
"""
import os
from glob import glob
from config import *
import numpy as np
import time
import process_audio
import matplotlib.pyplot as plt
from utilities import HzToNote, in_group, disp_spec

def ground_truth(song_root, max_length):
    """
    TODO:
        - rename max_length
        - note_range should be 88 not 108, and given by noteToHz tools
    Returns output as frequency-time numpy ndarray
    Parameters
    ----------
    song_root : string
                name of song without postfix (i.e. `.wav`)
    length : int
             maximum length of the output array, should be the same length of
             the input spectrogram
    Returns piano roll ndarray with shape (note_range, max_length)
    """
    sol_file = '{}.txt'.format(song_root)
    note_list = np.loadtxt(sol_file, skiprows=1)
    step_sec = time_step / sample_rate
    piano_roll = np.zeros((note_range, max_length)) # Change note range here
    if note_list.ndim == 1:
        note_list = note_list.reshape(1, -1)
    for [onset_time, offset_time, pitch] in note_list:
        # Onset and offset time in seconds
        onset_time_sec = float(onset_time)
        offset_time_sec = float(offset_time)
        pitch = int(pitch)
        # Decode note into midi roll at the time determined by the time_stemp
        # and onset/offset time given in quantized time-steps
        onset_time = int(np.floor(onset_time_sec / step_sec))
        offset_time = int(np.ceil(offset_time_sec / step_sec)) + 1
        # Between the onset and offset, we want a 1 since the note is being
        # played in the last frame

        # These (min)s are just so that we don't run into errors, and we'll
        # almost always choose the former (unless, for example, the note is too
        # high and beyond our spectrum)
        piano_roll[min(pitch, note_range - 1), min(onset_time, max_length - 1) : min(offset_time, max_length)] = 1
    return np.transpose(piano_roll)

def out_window(specgram):
    num_windows = specgram.shape[0]
    specgram = np.pad(specgram, ((0, 1), (0, 0)), "constant")
    windows = []
    for i in range(num_windows):
        w = specgram[i:i+1,:]
        windows.append(w)
    return np.array(windows)

def get_window_data_len(song_root):
    rate, data = process_audio.wav_root_to_data(song_root)
    if not rate == sample_rate:
        # We'll never really fall in here
        # Since scipy.io.wavfile.read always
        # Uses a rate of 44.1 kHz but just in case
        raise ValueError("rate {} does not equal hyper-parameter sample_rate, {}".format(rate, sample_rate))
    specgram = process_audio.constant_q_transform(data, preprocess_kernel, sample_rate, time_step)

def input_windows(song_root, preprocess_kernel):
    """
    Returns Windowed Frames of the CQT spectrogram
    Parameters
    ----------
    song_root : string
                name of song without postfix (i.e. `.wav`)
    preprocess_kernel : ndarray
                        CQT/STFT kernel
    Returns array of windows with shape (time_frames, window_size, frequencies)
    """
    rate, data = process_audio.wav_root_to_data(song_root)
    if not rate == sample_rate:
        # We'll never really fall in here
        # Since scipy.io.wavfile.read always
        # Uses a rate of 44.1 kHz but just in case
        raise ValueError("rate {} does not equal hyper-parameter sample_rate, {}".format(rate, sample_rate))
    specgram = process_audio.constant_q_transform(data, preprocess_kernel, sample_rate, time_step)
    specgram = np.transpose(specgram)
    num_windows = specgram.shape[0]
    specgram = np.pad(specgram, ((window_size // 2, window_size // 2), (0, 0)), "constant")
    windows = []
    for i in range(num_windows):
        w = specgram[i:i+window_size,:]
        windows.append(w)
    return np.array(windows)

class DataGen:
    def __init__(self, verbose=True):
        self.group_idx = 0
        self.group_n = 0
        self.groups = {}
        self.preprocess_kernel = process_audio.init_cqt(bins_per_octave, sample_rate, thresh, verbose=verbose)
        self.size = 0
        self.batch_size = batch_size
        self.steps_per_epoch = 0
        self.batch_queue = []
        self.initialized = False
        self.epochs = 1
        self.x = 0
        for name in training_names:
            self.groups[name] = []

    def add_items(self, group, items):
        if isinstance(items, list):
            self.groups[group].extend(items)
            self.size += len(items)
        else:
            self.groups[group].append(items)
            self.size += 1

    def init(self):
        """Run this after adding all items"""
        # Size up everything with output size
        self.steps_per_epoch = int(np.ceil(self.size / self.batch_size))
        self.initialized = True
        pass

    def __len__(self):
        return self.size - 1

    def __repr__(self):
        group_process_counts = ""
        for name in self.groups:
            group_process_counts += "{}: {}\n".format(name, len(self.groups[name]))
        if self.initialized:
            group_process_counts += "size: {}\nbatch size: {}\nsteps per epoch: {}".format(self.size, self.batch_size, self.steps_per_epoch)
        return group_process_counts

    def __iter__(self):
        return self

    def batch_next_song(self):
        next_found = True
        batch = []
        while self.group_n >= len(self.groups[training_names[self.group_idx]]):
            # This means we've hit the end of the group, and should keep moving
            # on to the next group until we're not at the end of a group
            # It's "while" because some groups in training_names may not exist
            # or be empty
            if self.group_idx < len(self.groups) - 1:
                self.group_idx += 1
                self.group_n = 0
            else:
                next_found = False
                break
        if next_found:
            # Return the element and increment group_n and r
            song_root = self.groups[training_names[self.group_idx]][self.group_n]
            self.group_n += 1
            # Run through loop for input and output
            """
            This is really bad code
            """
            inputs = input_windows(song_root, self.preprocess_kernel)
            #outputs = out_window(ground_truth(song_root, inputs.shape[0]))
            outputs = ground_truth(song_root, inputs.shape[0])
            input_num_windows = inputs.shape[0]
            #print("Initial Input Specgram Shape: {}".format(input_specgram.shape))
            #input_specgram = input_specgram.transpose()
            #print("Transposed Input Specgram Shape: {}".format(input_specgram.shape))
            #input_windows = window(input_specgram)
            #print("Input Windowed Shape: {}".format(input_windows.shape))
            #input_num_windows = input_windows.shape[0]
            #display_spectrogram(specgram, len(data), sample_rate, time_step)
            #####output_windows = window(output_specgram)
            #display_spectrogram(output, len(data), sample_rate, time_step)
            #input = input.reshape(1, -1)
            #####output_num_windows = len(output_windows)
            #print("Input Number of Windows: {}".format(input_num_windows))
            #print("Output Number of Windows: {}".format(output_num_windows))
            #rate, data = process_audio.wav_root_to_data(song_root)
            #print(len(data))
            #window_data_len = len(data) / input_num_windows
            ######################for idx in range(input_num_windows):
                #print("shape is {}".format(inputs[idx].shape))
                #print("num windows {}".format(input_num_windows))
                #disp_spec(np.transpose(inputs[idx]), window_data_len, sample_rate, time_step)
                #disp_spec(np.transpose(outputs[idx]), window_data_len, sample_rate, time_step)
                ################batch.append((np.array([inputs[idx]]), outputs[idx]))
                #print("INPSHAPE: {}\nOUTSHAPE: {}".format(input[idx].shape, output[idx].shape))
            #print("Input shape is {}".format(input_windows.shape))
            #print("Output shape is {}".format(output_specgram.shape))
            #print("INPSHAPE: {}\nOUTSHAPE: {}".format(inputs.shape, outputs.shape))
            batch.append((inputs, outputs))
            return batch
        return False

    def __next__(self):
        #print("Current x for either test/val: {}".format(self.x))
        self.x += 1
        if not self.initialized:
            self.init()
        batch = []
        while len(self.batch_queue) < batch_size:
            # Keep adding to the queue with the next
            # song's batch until it's full enough
            next_batch = self.batch_next_song()
            if not next_batch:
                if self.epochs <= epochs + 1: # extra space in case, TODO: +1 should be deleted
                    # regenerate for next epoch
                    print("Starting Next Epoch!")
                    print("Epoch count: {}".format(self.epochs))
                    self.group_idx = 0
                    self.group_n = 0
                else:
                    raise StopIteration()
            else:
                self.batch_queue.extend(next_batch)
            #print("Batch queue has length: {}".format(len(self.batch_queue)))
        batch = self.batch_queue[:batch_size]
        self.batch_queue = self.batch_queue[batch_size:]
        if len(batch) > 0: # this is maybe unnecessary
            # Convert batch from
            # [(x1, y1), (x2, y2), ...]
            # to
            # ([x1, x2, ..], [y1, y2, ...])
            inputs = []
            outputs = []
            for pair in batch:
                inputs.append(pair[0])
                outputs.append(pair[1])
            #print(len(inputs))
            #print(len(outputs))
            #print("input shape {} out shape {}".format(inputs[0].shape, outputs[0].shape))
            return inputs, outputs
            #return inputs[0], outputs[0]
        raise StopIteration()


class MAPS:
    def __init__(self, mus_path, verbose=False, super_verbose=False):
        t = time.time()
        self.train_gen = DataGen()
        self.val_gen = DataGen()
        self.test_gen = DataGen()
        self.unprocessed_count = 0
        self.processed_count = 0
        for root, dirs, files in os.walk(mus_path):
            temp_group = []
            temp_group_name = None
            for wav_name in glob(os.path.join(root, "*.wav")):
                song_root = wav_name.replace('.wav', '')
                found = False
                for name in training_names: # should be group_name to differentiate
                   if in_group(wav_name, name, pianos=tr_pianos):
                       temp_group.append(song_root)
                       if temp_group_name == None:
                           temp_group_name = name
                       if not name == temp_group_name:
                           raise ValueError("File System Error")
                       found = True
                if found:
                    self.processed_count += 1
                else:
                    self.unprocessed_count += 1
            if not temp_group_name == None:
                temp_len = len(temp_group)
                for x, i in enumerate([(self.train_gen, train_percent), (self.val_gen, val_percent), (self.test_gen, test_percent)]):
                    if len(temp_group) > 0:
                        # Add group to generator
                        if x < 2:
                            gen_group = temp_group[0:int(temp_len*i[1])]
                            temp_group = temp_group[int(temp_len*i[1]):]
                        else:
                            gen_group = temp_group
                        i[0].add_items(temp_group_name, gen_group)
        self.train_gen.init()
        self.val_gen.init()
        self.test_gen.init()

        if verbose:
            print("Initialized MAPS dataset class in {} seconds".format(time.time() - t))
        if super_verbose:
            print("MAPS Class: \n{}".format(self.__repr__()))
            print("----------------------------")
            gens = {"Training Generator": self.train_gen,
                    "Validation Generator": self.val_gen,
                    "Testing Generator": self.test_gen}
            for gen_name in gens:
                print("{}: \n{}".format(gen_name, gens[gen_name].__repr__()))
                print("----------------------------")
