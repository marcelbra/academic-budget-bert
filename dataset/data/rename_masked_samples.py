import os
import random

path = "/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/5_MaskedSamples/"
files = os.listdir(path)
random.shuffle(files)
n = len(files)
counter = 0
test_counter = 0
training_max = int(n*0.9)
for file in files:
    if counter < training_max:
        new_name = f"train_shard_{counter}.hdf5"
    else:
        new_name = f"test_shard_{test_counter}.hdf5"
        test_counter += 1
    os.rename(path + file, path + new_name)
    counter += 1
