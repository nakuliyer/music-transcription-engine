from utilities import noteToHz
import numpy as np

# CQT hyper-parameters
bins_per_octave = 24
sample_rate = 44100
thresh = 0.1
time_step = 2000

# Data generator
batch_size =  1 #16 # this apparently doesn't work
training_names = [
    "MAPS_ISOL_NO",
    "MAPS_ISOL_LG",
    "MAPS_ISOL_ST",
    "MAPS_ISOL_RE",
    "MAPS_ISOL_CH",
    "MAPS_ISOL_TR",
    "MAPS_RAND",
    "MAPS_UCHO",
    "MAPS_MUS"
]
ta_pianos = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb"]
tg_pianos = ["StbgTGd2", "SptkBGAm", "SptkBGCl"]
tr_pianos = ["ENSTDkAm", "ENSTDkCl"]
# percentages of dataset
train_percent = 0.85
test_percent = 0.1
val_percent = 0.05
training_steps_per_epoch = 1100
train_spe = train_percent * training_steps_per_epoch
test_spe = test_percent * training_steps_per_epoch
val_spe = val_percent * training_steps_per_epoch

# Training
epochs = 100
window_size = 7
input_max_frames = 100 # approximate memory cap on my machine
alpha = 0.001 # learning rate
beta_1 = 0.9
beta_2 = 0.999
momentum = 0.9 # SGD oscillation dampening factor
decay = 0.0 # rate of change in learning rate

fmin = noteToHz("A0")
fmax = noteToHz("C8")
note_range = 88 # should be dependent on other 2
midi_min = 21
midi_max = 108
num_freqs = int(np.ceil(bins_per_octave*np.log2(fmax/fmin)))

# output of prediction parsing
output_mask_threshold = 0.1
