import sys
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import datasets
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the raw dataset."
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output path to the raw dataset."
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
        help="The amount of split to process the dataset by."
    )
    args = parser.parse_args()
    return args

def process_wiki_from_disk():

    args = get_args()
    wiki = load_dataset("wikipedia", "20200501.en", split='train')
    #wiki = wiki.shard(num_shards=1000, index=0) # For testing
    path = args.dir
    out = args.o
    splits = args.splits
    n = len(wiki)
    portion = int(n / splits)

    # Create directories
    if not os.path.exists(out):
        os.mkdir(out)
    for i in range(splits):
        dir_name = out + str(i)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    for article_index, article in enumerate(tqdm(wiki)):
        article = article["text"]
        i = int(article_index//portion)
        if i!=splits:
            with open(f"{out}{i}/wiki_for_24h_BERT_{i}.txt", mode="a") as file:
                file.write(article.replace('\n', ' ') + "\n\n")

process_wiki_from_disk()