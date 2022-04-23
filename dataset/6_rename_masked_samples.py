import os
import random

path = "/mounts/data/proj/braasch/5_MaskedSamples/
#path = "/Users/marcelbraasch/Desktop/TestFiles/"
files = os.listdir(path)
random.shuffle(files)
n = len(files)
counter = 0
training_max = int(n*0.9)
for file in files:
    if counter < training_max:
        new_name = f"train_{counter}.hdf5"
    else:
        new_name = f"test_{counter}.hdf5"
    counter += 1
    os.rename(path + file, path + new_name)
