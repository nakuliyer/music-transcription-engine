"""
Methods for loading and manipulating dataset

author: Nakul Iyer
date: 2/28/19
"""
import os
from glob import glob
import numpy as np
import time

import process_audio
from utilities import Hz_to_note, in_group, disp_spec
from config import *

def ground_truth(song_root, max_length):
    """
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
        if midi_min <= pitch and pitch <= midi_max:
            piano_roll[pitch - midi_min, min(onset_time, max_length - 1) : min(offset_time, max_length)] = 1
    return np.transpose(piano_roll)

def input_windows(specgram):
    """
    Returns Windowed Frames of the CQT spectrogram
    Parameters
    ----------
    specgram: ndarray
              (frequency, time) spectrogram
    Returns array of windows with shape (time_frames, window_size, frequencies)
    """
    specgram = np.transpose(specgram)
    num_windows = specgram.shape[0]
    specgram = np.pad(specgram, ((window_size // 2, window_size // 2), (0, 0)), "constant")
    windows = []
    for i in range(num_windows):
        w = specgram[i:i+window_size,:]
        windows.append(w)
    return np.array(windows)

class DataGen:
    def __init__(self, type=None, verbose=True, use_batch=False, only_inputs=False):
        self.only_inputs = only_inputs
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
        self.use_batch = False
        self.type = type
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
        """Run this after adding all items if using batch"""
        if self.use_batch:
            self.steps_per_epoch = int(np.ceil(self.size / self.batch_size))
        self.initialized = True

    def __len__(self):
        if self.use_batch:
            return self.steps_per_epoch
        return 0

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

            # Get the input windowed 3D numpy array
            # Shape is (input_num_windows, window_length, frequencies)
            rate, data = process_audio.wav_root_to_data(song_root)
            if not rate == sample_rate:
                # We'll never really fall in here
                # Since scipy.io.wavfile.read always
                # Uses a rate of 44.1 kHz but just in case
                raise ValueError("rate {} does not equal hyper-parameter sample_rate, {}".format(rate, sample_rate))
            specgram = process_audio.constant_q_transform(data, self.preprocess_kernel, sample_rate, time_step)
            inputs = input_windows(specgram)
            input_num_windows = inputs.shape[0]

            # Get the output frame-by-frame 2D numpy array
            # Shape is (input_num_windows, midi_frequencies)
            outputs = ground_truth(song_root, input_num_windows)

            # Split the song into smaller windows if it is too large
            if input_num_windows >= input_max_frames:
                i = 0
                while (i + 1) * input_max_frames < input_num_windows:
                    input_case = inputs[input_max_frames * i: min(input_max_frames * (i+1), len(inputs) - 1)]
                    output_case = outputs[input_max_frames * i: min(input_max_frames * (i+1), len(inputs) - 1)]
                    batch.append((input_case, output_case))
                    i += 1
            else:
                batch.append((inputs, outputs))
            return batch
        return False

    def __next__(self):
        if not self.initialized:
            self.init()
        if self.type == "val":
            print("Validation")
        batch = []
        while len(self.batch_queue) < batch_size:
            # Keep adding to the queue with the next
            # song's batch until it's full enough
            next_batch = self.batch_next_song()
            if not next_batch:
                if self.epochs <= epochs:
                    # regenerate for next epoch
                    print("Starting Next Epoch! Count: {}".format(self.epochs))
                    self.group_idx = 0
                    self.group_n = 0
                    self.epochs += 1
                else:
                    raise StopIteration()
            else:
                self.batch_queue.extend(next_batch)
        batch = self.batch_queue[:batch_size]
        self.batch_queue = self.batch_queue[batch_size:]
        if len(batch) > 0:
            # Convert batch from
            # [(x1, y1), (x2, y2), ...]
            # to
            # ([x1, x2, ..], [y1, y2, ...])
            inputs = []
            outputs = []
            for pair in batch:
                inputs.append(pair[0])
                outputs.append(pair[1])
            if self.only_inputs:
                return inputs
            return inputs, outputs
        raise StopIteration()


class MAPS:
    def __init__(self, mus_path, verbose=False, super_verbose=False, only_inputs_test=False):
        t = time.time()
        self.train_gen = DataGen()
        self.val_gen = DataGen(type="val")
        self.test_gen = DataGen(type="train", only_inputs=only_inputs_test)
        self.unprocessed_count = 0
        self.processed_count = 0
        for root, dirs, files in os.walk(mus_path):
            temp_group = []
            temp_group_name = None
            for wav_name in glob(os.path.join(root, "*.wav")):
                song_root = wav_name.replace('.wav', '')
                found = False
                for name in training_names: # should be group_name to differentiate
                   if in_group(wav_name, name):
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
            print(self.__repr__())
            print("----------------------------")
            gens = {"Training Generator": self.train_gen,
                    "Validation Generator": self.val_gen,
                    "Testing Generator": self.test_gen}
            for gen_name in gens:
                print("{}: \n{}".format(gen_name, gens[gen_name].__repr__()))
                print("----------------------------")

    def __repr__(self):
        return "MAPS Dataset Class with {} processed files and {} unprocessed files".format(self.processed_count, self.unprocessed_count)
