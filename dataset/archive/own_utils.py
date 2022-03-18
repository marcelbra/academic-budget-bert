import pickle

from datasets import load_from_disk
from tqdm import tqdm
import os
from random import random
import nltk
import datasets
from math import sqrt
import matplotlib.pyplot as plt
import sys

def process_wiki_from_disk():
    sys.argv[2]
    path = "/media/marcelbraasch/Data2/DroppedDataset/Raw/"
    wiki = load_from_disk(path)
    n = len(wiki)
    half = n / 2
    i = 0
    for article in tqdm(wiki):
        article = article["text"]
        if i < half:
            with open("/media/marcelbraasch/Data2/DroppedDataset/Split/wiki_for_24h_BERT_1.txt", mode="a") as ofile1:
                ofile1.write(article.replace('\n', ' ') + "\n\n")
        else:
            with open("/media/marcelbraasch/Data2/DroppedDataset/Split/wiki_for_24h_BERT_2.txt", mode="a") as ofile2:
                ofile2.write(article.replace('\n', ' ') + "\n\n")
        i += 1

def compute_lexical_ratio():
    print("Preparing helper.")
    wiki = load_from_disk("/media/marcelbraasch/data/raw")["train"]
    half = int(len(wiki)/2)
    c = 0
    nltk.download("punkt")
    tokenizer = nltk.tokenize.word_tokenize
    closed_class_words = []
    with open("/dataset/helper/closed_class_words.txt") as f:
        for line in f:
            closed_class_words.extend(line.split())

    print("Creating lexical scores.")
    scores = []
    for article in tqdm(wiki):
        text = article["text"]
        words = tokenizer(text)
        n = len(words)
        closed_class_word_count = sum([1 for word in words if word in closed_class_words])
        ratio = closed_class_word_count / n
        scores.append(ratio)
    print("Adding.")
    wiki = wiki.add_column("LexicalRatio", scores)
    print("Sorting.")
    wiki.sort("LexicalRatio")
    print("Saving.")
    wiki.save_to_disk("/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/helper/Wikipedia/LexicalScores")



def plot_scores_and_lengths():

    data = []
    with open("../scores_and_lengths.pkl", "rb") as f:
        data = pickle.load(f)
    _, lengths = data
    x, y, = scores, lengths
    n = len(lengths)


    fig, ax = plt.subplots()
    ax.bar(list(range(len(x))), y, color='Red')
    ax.set_xticks(list(range(len(x))))
    # ax.set_xticklabels(x, rotation=35, ha='right', size=10)
    fig.tight_layout()
    plt.show()

    plot_doc_lengths_vs_score = False
    if plot_doc_lengths_vs_score:
        x = scores#[int(n*0.20):int(n*0.95)]
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.hist(x,bins=10000)
        ax.set_xlabel('Lexical Ratio')
        plt.show()

    if False:
        counter = 0
        for length in lengths:
            counter += 1
            if length >= 0.18:
                break
        percentage = counter / n
        print(percentage)


def prepare_dataset():
    path = "/dataset/helper/Wikipedia/LexicalScores"
    wiki = load_from_disk(path)
    wiki = wiki.sort("LexicalRatio")
    drop_ratio = 0.15
    for article in tqdm(wiki):
        if c < half:
            with open("/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/helper/Wikipedia/wiki_for_24h_BERT_1.txt", mode="w") as ofile1:
                ofile1.write(article["text"].replace('\n', ' ') + "\n\n")
        else:
            with open("/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/helper/Wikipedia/wiki_for_24h_BERT_2.txt", mode="w") as ofile2:
                ofile2.write(article["text"].replace('\n', ' ') + "\n\n")
        c += 1
    print("End writing.")
    print("End process")

def merge_shards():
    """
    Expects a (possibly recursively variably deep, in this case two) directory of
    model and test shards and renames them according to a counter.
    """
    files1 = os.listdir("/dataset/helper/Wikipedia/shards_1_copy")
    files2 = os.listdir("/dataset/helper/Wikipedia/shards_2_copy")
    training_counter, test_counter = 0,0
    for file in files1 + files2:
        if file.startswith("model"):
            os.rename(file, f"model{training_counter}.txt")
            training_counter += 1
        else:
            os.rename(file, f"test{test_counter}.txt")
            test_counter += 1

def rename_shards():
    path = "/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/helper/Wikipedia/masked_samples/"
    files = os.listdir(path)
    for file in files:
        prefix = "train" if random() < 0.8 else "test"
        new_name = file.split("_")[-1]
        os.rename(path + file, path + f"{prefix}_shard_{new_name}")

def avg_word_len_over_all_docs_distro():
    path = "/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/helper/Wikipedia/shards/"
    files = os.listdir(path)
    for file in tqdm(files):
        documents = []
        with open(path + file, "r") as f:
            documents = []
            word_lengths = 0
            word_amounts = 0
            i = 0
            for line in f:
                i += 1
                if line == "\n": # end of doc
                    documents.append(word_lengths / word_amounts)
                    word_lengths = 0
                    word_amounts = 0
                line = line[:-1]
                words = line.split()
                word_lengths += sum([len(x) for x in words])
                word_amounts += len(words)
            s = 0










