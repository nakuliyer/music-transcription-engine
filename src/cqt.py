import numpy as np
import os
from utilities import noteToHz
from scipy.sparse import csc_matrix
import sys
import matplotlib.pyplot as plt

def reduce_spec(specgram):
    return 20*np.log10(specgram)

class CQT:
    def __init__(self,
                 filters_per_octave,
                 sample_rate,
                 thresh,
                 fmin=None,
                 fmax=None):
        """
        Initializes the main kernel function using the Constant Q Transformation
        """
        if fmin == None:
            fmin = noteToHz("A1")
        if fmax == None:
            fmax = noteToHz("A7")
        self.filters_per_octave = filters_per_octave
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.thresh = thresh

        Q = 1.0/(2.0**(1.0/filters_per_octave) - 1)
        num_freqs = int(np.ceil(filters_per_octave*np.log2(fmax/fmin)))

        fft_len = int(2.0**np.ceil(np.log2(Q*sample_rate/fmin)))

        main_kernel = np.zeros((num_freqs, fft_len), dtype=complex)

        for freq_idx in range(num_freqs):
            freq = fmin*2.0**((freq_idx-1.0)/filters_per_octave) # maybe this can be in utilities... maybe its already there
            wnd_len = np.ceil(Q*sample_rate/freq)
            temp_ker = (np.hamming(wnd_len)/wnd_len)*np.exp(2*np.pi*(1j)*Q*np.arange(wnd_len)/wnd_len)
            spec_ker = np.fft.fft(temp_ker, fft_len)
            main_kernel[freq_idx, :] = spec_ker

        main_kernel[np.abs(main_kernel) <= thresh] = 0 # we can effects of this easily
        main_kernel = np.conjugate(csc_matrix(main_kernel)) / fft_len

        self.main_kernel = main_kernel
        self.num_freqs = num_freqs
        self.fft_len = fft_len

    def save_cqt_kernel():
        pass

    def load_cqt_kernel():
        pass

    def specgram(self, data, sample_rate, time_step):
        """
        """
        num_time_frames = int(np.ceil(len(data)/time_step)-1)
        #print("Padding Data")
        data = np.pad(data, (int(np.ceil(abs(self.fft_len - time_step) / 2)),
                                          int(np.floor(abs(self.fft_len - time_step) / 2))), 'constant',
                              constant_values=(0, 0))

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

    def disp_spec(self, data, sample_rate, time_step):
        """
        """
        specgram = self.specgram(data, sample_rate, time_step)
        plt.imshow(reduce_spec(specgram), aspect='auto', cmap='jet', origin='lower')
        plt.title('CQT spectrogram')
        plt.xticks(np.round(np.arange(1, np.floor(len(data)/sample_rate)+1)*sample_rate/time_step),
                   np.arange(1, int(np.floor(len(data)/sample_rate))+1))
        plt.xlabel('Time (s)')
        # Use noteToHz here for yticks
        plt.yticks(np.arange(1, 6*12*24/12+1, 12*24/12),
                ('A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'))
        plt.ylabel('Frequency (semitones/log(Hz))')
        plt.show()
