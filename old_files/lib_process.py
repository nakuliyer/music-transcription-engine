"""
This will hopefully be replaced by process_audio, which is my own
implementation of these methods where I have more control with the
algorithm myself. For now, though, I just want the alg to work
so I can feed the CNN.
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    return librosa.load(path)

def cqt(sr):
    C = np.abs(librosa.cqt(y, sr=sr))
    return C

def specgram(C):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(C,
                                                     ref=np.max),
                             sr=sr,
                             y_axis='mel',
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

y, sr = load("../sample_data/odetojoy.wav")
C = cqt(sr)
specgram(C)
