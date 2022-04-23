import random
import pickle
from copy import deepcopy
from math import log
import h5py

def create_word_from_indices(tokens, indices):
    word = ""
    for index in indices:
        curr_token = tokens[index]
        curr_token = curr_token[2:] if curr_token.startswith("##") else curr_token
        word += curr_token
    return word

def pmi(first_word, second_word):
    coocurence = cooc[frozenset({first_word, second_word})]
    first_freq, second_freq = freq[second_word], freq[second_word]
    return log(coocurence/(first_freq*second_freq)) if coocurence else float("-inf")

def pmi_masking(tokens, x):
    old_indices = deepcopy(x)
    new_indices = []
    while old_indices:
        first_word_indices = old_indices.pop(random.randrange(len(old_indices)))
        new_indices.append(first_word_indices)
        first_word = create_word_from_indices(tokens, first_word_indices)
        if not old_indices:
            return new_indices
        best_pmi = float("-inf")
        best_second_indices = None
        for second_word_indices in old_indices:
            second_word = create_word_from_indices(tokens, second_word_indices)
            if first_word == second_word: continue
            try:
                current_pmi = pmi(first_word, second_word)
            except:
                continue
            if current_pmi > best_pmi:
                best_second_indices = second_word_indices
                best_pmi = current_pmi
        if not best_second_indices is None:
            old_indices.remove(best_second_indices)
            new_indices.append(best_second_indices)
    return new_indices

def pmi_test(test_tokens, test_ids, targets):
    target_word = create_word_from_indices(test_tokens, targets)
    saved = []
    for indices in test_ids:
        word = create_word_from_indices(test_tokens, indices)
        try:
            current_pmi = pmi(target_word, word)
        except:
            print(word)
        saved.append((word, current_pmi))
    saved.sort(key=lambda x: x[1], reverse=True)
    print(*saved, sep=" ")


def wwm(tokens):
    indices = []
    current_index = []
    for i, token in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        current_token = tokens[i]
        next_token = tokens[i+1]
        current_index.append(i)
        if (not next_token.startswith("##")
                or (current_token.startswith("##")
                    and not next_token.startswith("##"))
        ):
            indices.append(current_index)
            current_index = []
    return indices

########################################################################################

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
cooc_path = "/mounts/work/kerem/Projects/pmi_masking/wiki_again/merge_final_from_merge_7/cooccurence.pickle"
word_path = '/mounts/work/kerem/Projects/pmi_masking/wiki_again/5_merged_data/vocab.pickle'
with open(cooc_path, "rb") as handle:
    cooc = pickle.load(handle)
with open(word_path, "rb") as handle:
    freq = pickle.load(handle)
text_to_tokens = lambda text: tokenizer.convert_ids_to_tokens((tokenizer.encode(text)))
test_ids = lambda tokens: wwm(tokens)

# article = "Define article here."
# test_tokens2 = text_to_tokens(article2)
# test_ids2 = wwm(test_tokens2)
# test_tokens3 = text_to_tokens(article3)
# test_ids3 = wwm(test_tokens3)
# test_tokens4 = text_to_tokens(article4)
# test_ids4 = wwm(test_tokens4)

########################################################################################

article1 = """Walt Disney's Wonderful World of Color.From 1963–65 he co-starred in the NBC legal drama Sam Benedict.O'Brien had a choice role in Seven Days in May (1964) which saw him receive a second Oscar nomination."I've never made any kind of personality success," he admitted in a 1963 interview."People never say 'that's an Eddie O'Brien part.'They say, 'That's a part Eddie O'Brien can play.'"  ""I'd like to be able to say something important," he added."To say something to people about their relationship with each other.If it touches just one guy, helps illustrate some points of view about living, then you've accomplished something."O'Brien worked steadily throughout the late 1960s and early 1970s.However his memory problems were beginning to take their toll.A heart attack meant he had to drop out of The Glass Bottom Boat (1966).Later career  "It would be awfully hard to do a series again," he said in a 1971 interview."I wouldn't go for an hour show again.They don't have much of a chance against the movies."He was a cast member of The Other Side of the Wind, Orson Welles' unfinished 1970s movie that finally was released"""
article2 = "to the television industry.Both were dedicated on February 8, 1960.Biography  Complete filmography  Partial television credits  Theatre  Hamlet (Oct 1936)  Daughters of Atreus (Oct 1936)  The Star Wagon (Sept 1937 – April 1938)  Julius Caesar (May 1938)  King Henry IV Part I (Jan–April 1939)  Leave Her to Heaven (Feb–March 1940)  Romeo and Juliet (May–June 1940)  Winged Victory (Nov 1943 – May 1944)  I've Got Sixpence (Dec 1952)  References  External links           Category:1915 births Category:1985 deaths Category:American male film actors Category:American male television actors Category:American male stage actors Category:American male radio actors Category:Best Supporting Actor Academy Award winners Category:Best Supporting Actor Golden Globe (film) winners Category:Burials at"
article3 = """he said in a 1971 interview."I wouldn't go for an hour show again.They don't have much of a chance against the movies."He was a cast member of The Other Side of the Wind, Orson Welles' unfinished 1970s movie that finally was released in 2018.In 1971, he was hospitalized with a "slight pulmonary condition."His last works"""
article4 = """The series was broadcast in three parts on BBC Two in December 2013.In three episodes Reeve travelled through Europe to the Holy Land in Israel.He retraced the route of ancient pilgrims.Tea Trail/Coffee Trail with Simon Reeve (2014) In The Tea Trail, Reeve travels from Mombasa, Kenya, and a tea auction, before taking the train to Nairobi and on into western Kenya visiting colonial plantations before crossing into Uganda, heading to Toro exploring the issue of child labour.On The Coffee Trail in Vietnam, Reeve heads south from Hanoi on the Reunification Express to Huế, where he visits the Khe Sanh Combat Base, before driving through coffee plantations to Buon Ma Thuot, meeting a coffee billionaire (Dang Le Nguyen Vu).He then also meets up with Dave D\'Haeze discussing the many problems of the growing"""

########################################################################################

path = "/mounts/data/proj/braasch/5_MaskedSamples/train_shard_100.hdf5"
path = "/home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/5_MaskedSamples_WWM100MASK/test_shard_1.hdf5"
f = h5py.File(path, "r")
input_ids, input_mask, masked_lm_ids, masked_lm_positions, segment_ids = [f[list(f.keys())[i]] for i in range(5)]
get_information = lambda x: print(f"input_ids[x] {input_ids[x]}", f"tokenizer.convert_ids_to_tokens(input_ids[x]) {tokenizer.convert_ids_to_tokens(input_ids[x])}", f"masked_lm_ids[x] {masked_lm_ids[x]}", f"masked tokens {tokenizer.convert_ids_to_tokens(masked_lm_ids[x])}",f"masked_lm_positions[x] {masked_lm_positions[x]}", f"Manually calculated mask position {[i for i,x in enumerate(input_ids[x]) if x==103]}", f"Difference {set(masked_lm_positions[x])-{i for i,x in enumerate(input_ids[x]) if x==103}}",sep="\n\n")

def tokens_unmasked(x):
    new_tokens = []
    masks = tokenizer.convert_ids_to_tokens(masked_lm_ids[x])
    positions = list(masked_lm_positions[x])
    for i,x in enumerate(tokenizer.convert_ids_to_tokens(input_ids[x])):
        if i in positions:
            token = masks[positions.index(i)]
        else:
            token = x
        new_tokens.append(token)
    return new_tokens

tokens_masked = lambda x: tokenizer.convert_ids_to_tokens(input_ids[x])
input_ids_masked = lambda x: input_ids[x]
input_ids_unmasked = lambda x: tokenizer.convert_tokens_to_ids(tokens_unmasked(x))

# Comparing masked tokens with unmasked input_ids
for i,j in zip(tokenizer.convert_ids_to_tokens(input_ids[15]), input_ids_unmasked(15)): print(i,j)

########################################################################################

gen_test_tokens = lambda x: tokens_unmasked(x)
gen_masked_tokens = lambda x: tokens_masked(x)

def gen_test_data_from_sample(x):
    test_tokens = gen_test_tokens(x)
    masked_tokens = gen_masked_tokens(x)
    test_tokens[0] = "[CLS]"
    test_ids = tokenizer.convert_tokens_to_ids(test_tokens)
    masked_positions = [i for i,x in enumerate(masked_tokens) if x=="[MASK]"]
    return test_tokens, masked_tokens, test_ids, masked_positions

def pmi_test2(test_tokens, test_ids, targets):
    target_word = create_word_from_indices(test_tokens, targets)
    other_words = tokenizer.decode(test_ids).split()
    saved = []
    for word in other_words:
        if word==target_word: continue
        try:
            current_pmi = pmi(target_word, word)
        except:
            print(word)
        saved.append((word, current_pmi))
    saved.sort(key=lambda x: x[1], reverse=True)
    print(saved[:10])

test_tokens, masked_tokens, test_ids, masked_positions = gen_test_data_from_sample(20)

for p in [[15], [18], [22], [43], [45], [48], [60], [62, 63], [81], [92]]:
    print(masked_tokens)
    print(test_tokens)
    print(pmi_test2(test_tokens,test_ids, p))
    print("\n"*3)
