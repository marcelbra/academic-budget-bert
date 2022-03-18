import os
from random import shuffle

path = "/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/5_MaskedSamples"
files = os.listdir(path)
shuffle(files)
n = len(files)
cut = n * 0.9
for index, file in enumerate(files):
    prefix = "train" if index + 1 <= cut else "test"
    os.rename(os.path.join(path, file), os.path.join(path, f"{prefix}_{index}"))
