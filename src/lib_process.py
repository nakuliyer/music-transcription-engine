import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    return librosa.load(path)

def cqt(sr):
    C = np.abs(librosa.cqt(y, sr=sr))

def specgram():
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
