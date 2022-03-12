import pickle
from datasets import load_dataset, load_from_disk
from nltk import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

def get_density(doc, closed_class_words, word_tokenizer):
	words = word_tokenizer(doc)
	count = sum([1 for word in words if word in closed_class_words])
	return 1 - count / len(words)

def get_closed_class_words():
	closed_class_words = []
	with open("closed_class_words.txt", "r") as f:
		for line in f:
			closed_class_words.extend(line.split())
	return closed_class_words

def lexical_density_sentences(wiki, closed_class_words):
    scores = []
    word_tokenizer = word_tokenize
    sent_tokenizer = sent_tokenize
    for article in tqdm(wiki):
        text = article["text"]
        sentences = sent_tokenize(text)
        densities = {"document_density": get_density(text, closed_class_words, word_tokenizer),
                     "sentence_densities": [get_density(sentence, closed_class_words, word_tokenizer)
                                            for sentence in sentences]}
        scores.append(densities)
    return scores

#wiki = load_dataset("wikipedia", "20200501.en")["train"]
closed_class_words = get_closed_class_words()

with open("densities.pkl", "rb") as handle:
    data = pickle.load(handle)


data.sort(key=lambda x: x["document_density"])
s = 0

# densities = lexical_density_sentences(wiki, closed_class_words)
# with open("densities.pkl", "wb") as handle:
#     pickle.dump(densities, handle, protocol=pickle.HIGHEST_PROTOCOL)




#
# def lexical_density_finetuning_ds(closed_class_words):
#     options = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
#     for option in options:
#         dataset = load_dataset("glue", option)
#         densities = []
#         for example in dataset["train"]:
#             if "sentence1" in example:
#                 sentence = example["sentence1"] + " " + example["sentence2"]
#             if "question1" in example:
#                 sentence = example["question1"] + " " + example["question2"]
#             if "premise" in example:
#                 sentence = example["premise"] + " " + example["hypothesis"]
#             if "question" in example:
#                 sentence = example["question"] + " " + example["sentence"]
#             if "sentence" in example:
#                 sentence = example["sentence"]
#             density = get_density(sentence, closed_class_words)
#             densities.append(density)
#         densities.sort()
#         plt.bar(list(range(len(densities))), densities)
#         plt.title(option + " " + "lexial_density")
#         plt.show()



