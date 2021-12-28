from datasets import load_from_disk
from tqdm import tqdm
import os

def process_wiki_from_disk():
    print("Start process")
    print("Start loading data.")
    wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")["train"]
    tenth = int(len(wiki)/10)
    print("End loading data.")
    c = 0
    print("Start writing.")
    with open("Wikipedia/wiki_for_24h_BERT_tenth.txt", mode="w") as ofile:
        for article in tqdm(wiki):
            ofile.write(article["text"].replace('\n', ' ') + "\n\n")
            c += 1
            if c>=tenth: break
    print("End writing.")
    print("End process")

def merge_shards():
    """
    Expects a (possibly recursively variably deep, in this case two) directory of
    training and test shards and renames them according to a counter.
    """
    files1 = os.listdir("~/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/shards_1")
    files2 = os.listdir("~/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/shards_2")
    training_counter, test_counter = 0,0
    for file in files1 + files2:
        if file.startswith("training"):
            os.rename(file, f"training{training_counter}.txt")
            training_counter += 1
        else:
            os.rename(file, f"training{test_counter}.txt")
            test_counter += 1

merge_shards()
