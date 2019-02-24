from config import *
from keras_network import Net
from dataset import MAPS

mus_path = "F:\\MAPS"

# Verbose means that initialization times and other
# Functional Time Elapsed information will be printed
verbose = True

# Super Verbose means that some output will be
# Printed, and should be used only for elementary
# testing purposes
super_verbose = True


maps = MAPS(mus_path, verbose=verbose, super_verbose=super_verbose, only_inputs_test=True)
net = Net(verbose=verbose, model="note_cnn", optimizer="adam", reload_model=False)

ex_test_file = "F:\\MAPS\\MAPS_ENSTDkAm_1\\ENSTDkAm\\ISOL\\NO\\MAPS_ISOL_NO_F_S0_M50_ENSTDkAm"
#ex_test_file = "F:\\MAPS\\MAPS_ENSTDkAm_2\\ENSTDkAm\\MUS\\MAPS_MUS-bk_xmas1_ENSTDkAm"
net.run_test(ex_test_file, verbose=True)
#net.run_test("F:\\MAPS\\MAPS_ENSTDkAm_2\\ENSTDkAm\\MUS\\MAPS_MUS-chpn_op25_e4_ENSTDkAm", verbose=True)
