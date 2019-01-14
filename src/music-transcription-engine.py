"""
"""
from dataset import MAPS
import process_audio
import lib_process

#mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_COMPLETE"
mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_SAMPLE"

maps = MAPS(mus_path)
x = 0
for song_root in maps:
    wav_name = song_root + ".wav"
    #process_audio.audioToData(wav_name)
    y, sr = lib_process.load(wav_name)
    C = lib_process.cqt(sr)
    if x % 100 == 0:
        print("{}% done loading dataset".format(x*100/len(maps)))
    x += 1
