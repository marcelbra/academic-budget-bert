import sys
from datasets import load_from_disk
from tqdm import tqdm
import datasets
import os

def process_wiki_from_disk():
    path = sys.argv[2]
    out = sys.argv[4]
    wiki = load_from_disk(path)
    n = len(wiki)
    half = n / 2
    i = 0
    if not os.path.exists(out):
        os.mkdir(out)
    if not os.path.exists(out + "First/"):
        os.mkdir(out + "First/")
    if not os.path.exists(out + "Second/"):
        os.mkdir(out + "Second/")
    for article in tqdm(wiki):
        article = article["text"]
        if i < half:
            with open(out + "First/" + "wiki_for_24h_BERT_1.txt", mode="a") as ofile1:
                ofile1.write(article.replace('\n', ' ') + "\n\n")
        else:
            with open(out + "Second/" + "wiki_for_24h_BERT_2.txt", mode="a") as ofile2:
                ofile2.write(article.replace('\n', ' ') + "\n\n")
        i += 1

process_wiki_from_disk()