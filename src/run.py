"""
This code is really really tentative and bad
"""

from keras.models import Model, load_model
from config import *
import process_audio
import time
import sys
import os
from dataset import input_windows
from keras_network import f1, loss
from utilities import disp_spec
import matplotlib.pyplot as plt

verbose = True
ex_test_file = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_SAMPLE/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M57_AkPnBcht"
#ex_test_file = "/Users/nakul/Documents/School/CSC 600/MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht"

model_name = "resnet_model"#--std--adam--binary_crossentropy" # tentative
model_loc = os.path.join("..", "stored_models", "{}.h5".format(model_name))
if os.path.isfile(model_loc):
    if verbose:
        print("Found Model at {}".format(model_loc))
    model = load_model(model_loc, custom_objects={"loss": loss, "f1": f1})
else:
    raise SystemError("No Model Found at {}".format(model_loc))

cqt = process_audio.init_cqt(bins_per_octave, sample_rate, thresh, verbose=verbose)

inputs = input_windows(ex_test_file, cqt)

S = model.predict(inputs, batch_size=batch_size, verbose=1)

#S = []

#for window in input:
#    S.append(model.predict(np.array([window]), batch_size=batch_size, verbose=1)[0])

#S = np.array(S)
print(S.shape)
rate, data = process_audio.wav_root_to_data(ex_test_file)
disp_spec(S.transpose(), len(data), sample_rate, time_step)
#for time_idx in range(len(S)):
    #print(S[time_idx])
#    time_sliced_thresh = S[time_idx] < 0.023
    #print(time_sliced_thresh)
#    S[time_idx][time_sliced_thresh] = 0
#    print(S[time_idx])
# need threshhold
#plt.plot(S)
#plt.show()
#rate, data = process_audio.wav_root_to_data(ex_test_file)
#disp_spec(S.transpose(), len(data), sample_rate, time_step)
