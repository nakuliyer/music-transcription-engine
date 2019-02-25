from config import *
from keras_network import Net
from dataset import MAPS
from sheet_music import piano_roll_to_sheet_music

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True

net = Net(verbose=verbose, model="std_gpu", optimizer="adam", reload_model=True, use_cuda=False)

#ex_test_file = "F:\\MAPS\\MAPS_ENSTDkAm_1\\ENSTDkAm\\ISOL\\NO\\MAPS_ISOL_NO_F_S0_M50_ENSTDkAm"
#save_location = "..\\output_saves\\MAPS_ISOL_NO_F_S0_M50_ENSTDkAm"
ex_test_file = "F:\\MAPS\\MAPS_ENSTDkAm_2\\ENSTDkAm\\MUS\\MAPS_MUS-bk_xmas1_ENSTDkAm"
save_location = "..\\output_saves\\MAPS_MUS-bk_xmas1_ENSTDkAm"

piano_roll = net.run_test(ex_test_file, verbose=verbose, super_verbose=super_verbose, output_avg_of=10)
piano_roll_to_sheet_music(piano_roll, save_location)
