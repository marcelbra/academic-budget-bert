import argparse
import os
from datasets import load_from_disk, load_dataset
from transformers import AutoModel, RobertaTokenizer, RobertaModel, Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk import sent_tokenize
import torch
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/0_LexicalScores",
        help="Path to the raw dataset."
    )
    parser.add_argument(
        "-o",
        type=str,
        default="/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia_Test/1_Dropped",
        help="Output path to the raw dataset."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Num workers."
    )
    args = parser.parse_args()
    return args

def load_wiki(path):
    try:
        wiki = load_from_disk(path)
    except:
        # Loading version not tested yet
        wiki = load_dataset("Wikipedia")
    return wiki

def load_cola_model(device, name="yoshitomo-matsubara/bert-base-uncased-cola"):
    tokenizer = AutoTokenizer.from_pretrained(name)#.to(device)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    return model, tokenizer

def make_prediction(model, tokenizer, device, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    logits = model(**inputs)[0][0][1]
    return float(logits) # >= 4.5

device = "cuda:0" if torch.cuda.is_available() else "cpu"
sent_tokenizer = sent_tokenize
model, tokenizer = load_cola_model(device)
tracker = defaultdict(lambda: defaultdict(int))

def filter_sentences(example):
    global tracker
    sentences = sent_tokenizer(example["text"])
    new_sentences = []
    for sentence in sentences:

        try:
            pred = make_prediction(model, tokenizer, device, sentence)# >= 4.5
        except:
            pred = None

        if pred is not None:
            # for portion in [3.5+x*0.1 for x in range(1, 15)]:
            #     if pred > portion:
            #         tracker[portion]["keep"] += 1
            #     else:
            #         tracker[portion]["drop"] += 1

            if pred > 5:
                new_sentences.append(sentence)

        else:

            for portion in [3.5+x*0.1 for x in range(1, 15)]:
                tracker[portion]["drop"] += 1

    if not new_sentences:
        example["text"] = "Empty."
    else:
        example["text"] = " ".join(new_sentences)
    return example



def create_new_dataset():

    args = get_args()
    wiki = load_wiki(args.dir)
    #wiki = wiki.shard(num_shards=60000, index=0)
    #n = len(wiki)
    # wiki = wiki.sort("LexicalRatio")
    #wiki = wiki.map(filter_sentences)
    #wiki = wiki.filter(lambda example: not example['text'].startswith("Empty."))
    # wiki = wiki.add_column"("Index", range(n))
    # wiki = wiki.filter(lambda x: x["Index"] >= int(n*0.1), num_proc=args.num_workers)
    # s = {x[0]:x[1]["drop"]/(x[1]["drop"]+x[1]["keep"]) for x in tracker.items()}
    if not os.path.exists(args.o):
        os.mkdir(args.o)
    #wiki = wiki.shuffle(seed=42)
    wiki.save_to_disk(args.o)

create_new_dataset()