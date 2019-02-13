"""
Helper Functions for processing audio
"""
import scipy.io.wavfile as wav
import soundfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from cqt import CQT
import sys
import pydub
import time
from config import *

def audioToData(audio_path):
    if audio_path.endswith(".wav"):
        return wav.read(audio_path)
    elif audio_path.endswith(".mp3"):
        sound = pydub.AudioSegment.from_mp3(audio_path)
        wv = sound.export(audio_path[:-4]+".wav", format="wav")
        return wav.read(audio_path[:-4]+".wav")
    sys.exit("Could not accept data. Try `.wav` file type for best results.")

def wav_root_to_data(song_root):
    return audioToData(song_root + ".wav")

def init_cqt(bins_per_octave, sample_rate, thresh, verbose=False):
    return CQT(bins_per_octave, sample_rate, thresh, fmin, fmax, verbose)

def constant_q_transform(data, cqt, sample_rate, time_step, image=False):
    data = data/(2.0**(data.itemsize*8-1))
    data = np.mean(data, 1)
    if image:
        return cqt.disp_spec(data, sample_rate, time_step)
    return cqt.specgram(data, sample_rate, time_step)

def onset_detect(specgram):
    f = specgram.sum(axis=0)
    spec_len = f.shape[0]
    print(spec_len)
    onset_map = np.zeros(spec_len)
    for m in range(spec_len):
        if m == 0:
            onset_map[m] = 1
        onset_map[m] = (f[m]-f[m-1])/f[m]
    plt.plot(onset_map)
    plt.show()
    return onset_map
