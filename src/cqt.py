import numpy as np
import os
from utilities import noteToHz
from scipy.sparse import csc_matrix
import sys
import matplotlib.pyplot as plt
import time
from config import *

def reduce_spec(specgram):
    result = np.nan_to_num(specgram)
    return 20*np.log10(result)

class CQT:
    def __init__(self,
                 bins_per_octave,
                 sample_rate,
                 thresh,
                 fmin,
                 fmax,
                 verbose=False):
        """
        Initializes the main kernel function using the Constant Q Transformation
        """
        t = time.time()
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.thresh = thresh

        # Constant Q = frequency to resolution ratio
        Q = 1.0/(2.0**(1.0/bins_per_octave) - 1)

        fft_len = int(2.0**np.ceil(np.log2(Q*sample_rate/fmin)))

        main_kernel = np.zeros((num_freqs, fft_len), dtype=complex)

        # X(k) = ∑ w(k, n) * x(n) * e^(− j2πQn/N(k)) from n = 0 to N(k) - 1 for k = 1, 2,..., N(p)

        for freq_idx in range(num_freqs):
            freq = fmin*2.0**((freq_idx-1.0)/bins_per_octave) # maybe this can be in utilities... maybe its already there
            wnd_len = np.ceil(Q*sample_rate/freq)
            temp_ker = (np.hamming(wnd_len)/wnd_len)*np.exp(2*np.pi*(1j)*Q*np.arange(wnd_len)/wnd_len)
            spec_ker = np.fft.fft(temp_ker, fft_len)
            main_kernel[freq_idx, :] = spec_ker

        #print("Main kernel shape: {}".format(main_kernel.shape))

        main_kernel[np.abs(main_kernel) <= thresh] = 0 #np.finfo(float).eps # we can effects of this easily
        main_kernel = np.conjugate(csc_matrix(main_kernel)) / fft_len

        self.main_kernel = main_kernel
        self.num_freqs = num_freqs
        self.fft_len = fft_len

        if verbose:
            print("Initialized CQT in {} seconds".format(time.time() - t))

    def save_cqt_kernel():
        pass

    def load_cqt_kernel():
        pass

    def specgram(self, data, sample_rate, time_step):
        """
        """
        num_time_frames = int(np.ceil(len(data)/time_step)-1)
        #print("Padding Data")
        #print("Shapedat {}".format(data.shape))
        data = np.pad(data, (int(np.ceil(abs(self.fft_len - time_step) / 2)),
                                          int(np.floor(abs(self.fft_len - time_step) / 2))), 'constant',
                              constant_values=(0, 0))
        #print("Post-up Shapedat {}".format(data.shape))
        #print("window_size: {}".format(window_size))

        # We're returning a spectrogram of size
        # Number of Frequencies by Number of Time Frames
        # (i.e. we've essentially quantized the freq-time space)
        specgram = np.zeros((self.num_freqs, num_time_frames))

        # Sparse because it is filled with 0s (has to meet energy threshold)
        sparse_ker = self.main_kernel
        fft_len = self.fft_len
        #print("Analyzing {} frames".format(num_time_frames))
        for t in range(num_time_frames):
            data_idx = t * time_step
            specgram[:, t] = abs(sparse_ker * np.fft.fft(data[data_idx:data_idx + fft_len]))
        return specgram

    def disp_spec(self, data, sample_rate, time_step): # unnecessary
        """
        """
        specgram, windows = self.specgram(data, sample_rate, time_step)
        plt.imshow(reduce_spec(specgram), aspect='auto', cmap='inferno', origin='lower')
        plt.title('CQT spectrogram')

        time_ticks = int(np.ceil(len(data)/sample_rate))

        plt.xticks(np.round(np.arange(1, time_ticks)*sample_rate/time_step),
                   np.arange(1, time_ticks))
        plt.xlabel('Time (s)')
        # Use noteToHz here for yticks
        # plt.yticks(np.arange(1, self.bins_per_octave * 6, self.bins_per_octave),
        #         ('A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'))
        plt.yticks(np.arange(1, self.bins_per_octave * 8, self.bins_per_octave),
                ('A1','A2','A3','A4','A5','A6', 'A7', 'A8'))
        plt.ylabel('Frequency (semitones/log(Hz))')
        plt.show()
        return specgram
