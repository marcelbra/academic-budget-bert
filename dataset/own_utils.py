from datasets import load_from_disk

def process_wiki_from_disk():
    wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")["train"]
    with open("Wikipedia/wiki_for_24h_BERT.txt", mode="w") as ofile:
        for article in wiki[:100]["text"]:
            ofile.write(article.replace('\n', ' ') + "\n\n")