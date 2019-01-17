"""
"""
import scipy.io.wavfile as wav
from scipy import signal
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import numpy as np
import cqt

def audioToData(audio_path):
    return wav.read(audio_path)

def dataToSpec(sameple_rate, samples):
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    print(spectrogram)
    print(spectrogram.shape)
    spectrogram = spectrogram.reshape(-1, 2)
    print(spectrogram)
    print(spectrogram.shape)

    #plt.pcolormesh(times, frequencies, spectrogram)
    plt.pcolormesh(spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

""" scale frequency axis logarithmically """
def log_freq_ax(spec, sr=44100, factor=20):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

def plotstft(samplerate, samples, binsize=2**10, plotpath=None, colormap="jet"):
    #samplerate, samples = wav.read(audiopath)
    s = cqt.cqt(samples)

    #sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(s)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

def cqt_spec(X, sr, bin_size, cmap="inferno"):
    pass

#sample_rate, samples = audioToData('../sample_data/unravel.wav')
#plotstft(sample_rate, samples)
#time = np.arange(len(samples[:,0]))*1.0/sample_rate

#plt.show()
