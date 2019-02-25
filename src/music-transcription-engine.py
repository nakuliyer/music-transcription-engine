"""
Runs the program
TODO:
1. experiment with stft instead of process_audio
2. integrate onset_detect with cqt
4. add example test.py files for everything
5. turn those examples easily into a ipynb
6. prettify the code (i.e. use two string quotes everywhere, etc)
7. The thing which finds stuff in the dataset should use regex (i.e. "MAPS_MUS\w+*")
"""
from dataset import MAPS, in_group
from config import *
import process_audio
import time
import numpy as np
from utilities import dl_spec
import keras_network as network
import sys
import os
import matplotlib.pyplot as plt

start = time.time()

# Path to the MAPS Dataset
mus_path = "F:\\MAPS"

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True

print("Welcome to Sheet Music Helper!")
maps = MAPS(mus_path, verbose=verbose, super_verbose=super_verbose)
net = network.Net(verbose=verbose, model="std_gpu", optimizer="adam", reload_model=False)

print("Ready for Training after {} seconds!".format(time.time() - start))
start = time.time()
net.train(maps.train_gen, maps.val_gen, verbose=verbose, super_verbose=super_verbose)
print("Training took a total of {} seconds!".format(time.time() - start))

net.evaluate(maps.test_gen, verbose=True)
