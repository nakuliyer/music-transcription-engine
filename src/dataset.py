"""
Loading and manipulating dataset
"""
import os
from glob import glob

class MAPS:
    def __init__(self, mus_path):
        self.files = [wav_name.replace('.wav', '') for root, dirs, files in os.walk(mus_path) for wav_name in glob(os.path.join(root, "*.wav"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.files[i]
