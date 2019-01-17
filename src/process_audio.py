"""
"""
import scipy.io.wavfile as wav
import soundfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from cqt import CQT
import sys
import pydub

def audioToData(audio_path):
    if audio_path.endswith(".wav"):
        return wav.read(audio_path)
    elif audio_path.endswith(".mp3"):
        sound = pydub.AudioSegment.from_mp3(audio_path)
        wv = sound.export(audio_path[:-4]+".wav", format="wav")
        return wav.read(audio_path[:-4]+".wav")
    sys.exit("Could not accept data. Try `.wav` file type for best results.")

def init_cqt(filters_per_octave, sample_rate, thresh):
    return CQT(filters_per_octave, sample_rate, thresh)

def constant_q_transform(data, cqt, sample_rate, time_step, image=False):
    # Do some audio normalization here if necessary
    if data.ndim > 1:
         data = data[:,0]
    if image:
        return cqt.disp_spec(data, sample_rate, time_step)
    return cqt.specgram(data, sample_rate, time_step)

# rate, data = audioToData('../sample_data/odetojoy.wav')
# if data.ndim > 1:
#     data = data[:,0]
# time = np.arange(len(data))*1.0/rate
#
# plt.plot(time,data)
# plt.show()
#
# print("The shape of the data is {}".format(data.shape))
# f, t, Zxx = signal.stft(data, fs=rate)
# print(f.shape)
# print("T is {}".format(t.shape))
# print(Zxx.shape)
# plt.pcolormesh(t, f, np.abs(Zxx))
# plt.show()
#
# spec_data = CQT().specgram(data) # TODO: Give cqt sample rate
#
# plt.imshow(spec_data, aspect='auto', cmap='inferno', origin='lower')
# plt.show()

# nfft = 2**12
# pxx, freq, bins, plot = plt.specgram(data, NFFT=nfft, Fs=2, cmap="inferno")
# plt.show()

#print("The shape post_cqt is {}".format(post_cqt.shape))
#f, t, Sxx = signal.spectrogram(post_cqt, rate)
#print(f.shape)
#print("New t is {}".format(t.shape))
#print(Sxx.shape)

#pxx, freq, bins, plot = plt.specgram(Sxx, NFFT=nfft, Fs=2, cmap="inferno")
#plt.show()
#plt.pcolormesh(t, f, np.abs(Sxx))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
