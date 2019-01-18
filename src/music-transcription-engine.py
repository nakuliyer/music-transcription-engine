"""
Runs the program
TODO:
- experiment with stft instead of process_audio
"""
from dataset import MAPS
import process_audio
import time
import nmf
import sys

#mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_COMPLETE"
mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_SAMPLE"
maps = MAPS(mus_path)

# CQT hyper-parameters
filters_per_octave = 24
sample_rate = 44100
thresh = 0.1
time_step = 2000
cqt = process_audio.init_cqt(filters_per_octave, sample_rate, thresh)
print("Initialized CQT kernel")

rate, data = process_audio.audioToData("../sample_data/unravel.wav")
specgram = process_audio.constant_q_transform(data, cqt, sample_rate, time_step, True)
print(specgram)
sys.exit()

x = 0
t = time.time()
for song_root in maps:
    #print("Analyzing song {}".format(song_root))
    wav_name = song_root + ".wav"
    rate, data = process_audio.audioToData(wav_name)
    #print("Converted Audio To Data")

    if not rate == sample_rate:
        # We'll never really fall in here
        # Since scipy.io.wavfile.read always
        # Uses a rate of 44.1 kHz but just in case
        sample_rate = rate
        cqt = process_audio.init_cqt(filters_per_octave, sample_rate, thresh)
    #print("Creating Spec")
    specgram = process_audio.constant_q_transform(data, cqt, sample_rate, time_step, True)
    print(nmf.nmf(specgram))
    if x % 100 == 0:
        print("{}% done loading dataset in {} secs".format(x*100/len(maps), time.time()-t))
    x += 1

    # Take out this break later on obviously
    break
