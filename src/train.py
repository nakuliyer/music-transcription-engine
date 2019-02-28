"""
Prepares Data and Trains Model
Entry point for training; please set mus_path, verbose, and super_verbose
appropriately

author: Nakul Iyer
date: 2/28/19
"""
import time

import process_audio
import network
from dataset import MAPS, in_group
from config import *

# Path to the MAPS Dataset
mus_path = "F:\\MAPS"

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True

start = time.time()
print("Welcome to Sheet Music Helper!")
maps = MAPS(mus_path, verbose=verbose, super_verbose=super_verbose)
net = network.Net(verbose=verbose, model="std_gpu", optimizer="adam", reload_model=False)

print("Ready for Training after {} seconds!".format(time.time() - start))
start = time.time()
net.train(maps.train_gen, maps.val_gen, verbose=verbose, super_verbose=super_verbose)
print("Training took a total of {} seconds!".format(time.time() - start))
net.evaluate(maps.test_gen, verbose=True)
