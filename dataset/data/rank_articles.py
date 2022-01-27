import pickle
import os
import operator
import nltk.tokenize
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from multiprocessing import Process, Manager


def rank():
    nltk.download("punkt")
    tokenizer = nltk.tokenize.word_tokenize
    closed_class_words = []
    with open("closed_class_words.txt") as f:
        for line in f:
            closed_class_words.extend(line.split())


    # TODO
    """
    
    Komplette Wikipedia einlesen und so wie unten verarbeiten.
    Wenn die Shard fertig ist soll die Liste mit (index, document, ratio)
    in einem file gespeichert werden.
    
    """

    path = "Wikipedia/shards_merged/"
    files = os.listdir(path)
    i, counter = 0, 0

    for file in tqdm(files):
        documents = []
        with open(path + file, "r") as f:
            document = ""
            for sentence in f:
                # end of document
                if sentence == "\n":
                    n = len(document)
                    words = tokenizer(document)
                    closed_class_word_count = sum([1 for word in words if word in closed_class_words])
                    ratio = closed_class_word_count / n
                    documents.append((i, document, ratio))
                    document = ""
                    i += 1
                document += sentence[:-1] + " "
            with open(f"Wikipedia/sorted/lexical_count_{counter}.pkl", "wb") as handle:
                pickle.dump(documents, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter += 1

rank()

"""
def f(lock, files, documents, closed_class_words):
    tokenizer = nltk.tokenize.word_tokenize
    for sentence in files:
        # end of document
        if sentence == "\n":
            n = len(document)
            words = tokenizer(document)
            closed_class_word_count = sum([1 for word in words if word in closed_class_words])
            ratio = closed_class_word_count / n
            with lock:
                documents.append((document, ratio))
            document = ""
        document += sentence[:-1] + " "
        
        
        

workers = 16

closed_class_words = []
with open("closed_class_words.txt") as f:
    for line in f:
        closed_class_words.extend(line.split())

path = "Wikipedia/shards/"
files = os.listdir(path)
shards = []
for file in tqdm(files):
    with open(path + file, "r") as f:
        shards.append(f.readlines())

index = []
amount_articles = 6000000
bucket_size = amount_articles / workers
for i in range(workers):
    index.append((int(i * bucket_size), int((i+1) * bucket_size)))

files = []

docs = []
with Manager() as manager:

    lock = manager.Lock()
    documents = manager.list()
    processes = []
    for i in range(workers):
        p = Process(target=f, args=(lock,
                                    files[index[i][0]: index[i][1]],
                                    documents,
                                    closed_class_words))
        processes.append(p)
    processes[0].start()
    #for p in processes:
    #    p.start()
    #for p in processes:
    #    p.join()

    #docs = documents

#documents.sort(key=operator.itemgetter(1))
#with open("Wikipedia/sorted/lexical_count_6.pkl", "wb") as handle:
#    pickle.dump(documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

#rank()
"""