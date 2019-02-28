import sys

import network
from dataset import MAPS
from sheet_music import piano_roll_to_sheet_music
from config import *

assert len(sys.argv) == 3, "No input and output location specified"

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True

net = network.Net(verbose=verbose, model="std_gpu", optimizer="adam", train_spe=935.0, reload_model=True, use_cuda=False)

ex_test_file = sys.argv[1][:-4] # Remove `.wav` ending
save_location = sys.argv[2][:-4] # Remove `.pdf` ending

piano_roll = net.run_test(ex_test_file, verbose=verbose, super_verbose=super_verbose, output_avg_of=10)
piano_roll_to_sheet_music(piano_roll, save_location)
