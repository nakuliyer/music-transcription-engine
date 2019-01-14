"""
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt

import os

print("Packages Loaded")

MUSIC_ROOT = '../sample_data' # Music Dataset (MAPS ~ 31GB) Directory

filename = 'unravel.wav'
X, sr = librosa.load(os.path.join(MUSIC_ROOT, filename))
print(X, sr)
print(X.shape)
print(sr)

print("Librosa Loaded")

# CQT hyperparameters
hop_length = 512 # samples between successive CQT columns
fmin = 32.7 # minimum frequency in Hz (low C ~ 32.7 Hz)
bins_per_octave = 32
n_bins = bins_per_octave * 8 # frequency bins starting at fmin (8 octaves)
filter_scale = 1
norm = 1

# cqt
def cqt(X, sr):
    cqt = librosa.core.cqt(X, sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, filter_scale=filter_scale, norm=norm)
    return np.abs(cqt)

def plot_(X, sr):
    y = cqt(X, sr)
    print(y.shape)
    plt.pcolormesh(y, cmap='inferno')
    plt.show()

plot_(X, sr)
