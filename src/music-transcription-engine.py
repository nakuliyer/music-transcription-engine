"""
Runs the program
TODO:
1. GPU!!!!!!!!!!!!
1. experiment with stft instead of process_audio
2. integrate onset_detect with cqt
3. add verbose to everything
4. add example test.py files for everything
5. turn those examples easily into a ipynb
6. prettify the code (i.e. use two string quotes everywhere, etc)
7. The thing which finds stuff in the dataset should use regex (i.e. "MAPS_MUS\w+*")
"""
from dataset import MAPS, in_group
from config import *
import process_audio
import time
#import nmf
import keras_network as network
import sys

# Path to the MAPS Dataset
mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS"
#mus_path = "/Volumes/CCCOMA_X64FRE_EN-US_DV9/MAPS_SAMPLE"
#mus_path = "/Users/nakul/Documents/School/CSC 600/MAPS_AkPnBcht_2"

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True

maps = MAPS(mus_path, verbose=verbose, super_verbose=super_verbose)
net = network.Net(verbose=verbose, optimizer="adam")

# print("Train Gen SPE: {}".format(len(maps.train_gen)))
# print("Validation Gen SPE: {}".format(len(maps.val_gen)))
# t = time.time()
# x = 0
# for i in maps.train_gen:
#     x += 1
#     #print("Input shape: {}".format(i[0].shape))
#     #print("Output shape: {}".format(i[1].shape))
#     if x % 10:
#         print("{}% done".format(x * 100))
#     break
# print("Generating took {} seconds".format(time.time() - t))
# print("Steps per Epoch is {}".format(x))
# sys.exit()

net.train(maps.train_gen, maps.val_gen, verbose=verbose, super_verbose=super_verbose)
net.evaluate(maps.test_gen, verbose=True)
