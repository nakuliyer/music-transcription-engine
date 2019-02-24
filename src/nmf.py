"""
Methods for NMF training and transcription
TODO:
1. Implement verbose graphs
"""
from sklearn.decomposition import NMF
from dataset import parse_solution
import process_audio
import numpy as np

def nmf_train_notes(maps, cqt, sample_rate, time_step):
    isolated_notes = maps.groups["ISOL_NO"]
    for song_root in isolated_notes:
        rate, data = process_audio.wav_root_to_data(song_root)

        if not rate == sample_rate:
            # We'll never really fall in here
            # Since scipy.io.wavfile.read always
            # Uses a rate of 44.1 kHz but just in case
            sample_rate = rate
            cqt = process_audio.init_cqt(bins_per_octave, sample_rate, thresh)

        specgram = process_audio.constant_q_transform(data, cqt, sample_rate, time_step)

    # Verbose graphs here
    pass

def nmf_train_music(song_root):
    pass

def nmf_transcribe(specgram):
    pass
