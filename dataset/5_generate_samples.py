# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from multiprocessing import Manager, Process
import pickle5 as pickle
import argparse
import collections
import os
import random
from io import open
from math import log
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import six

logger = logging.getLogger()

# from dataset.helper.utils import convert_to_unicode
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class TrainingInstance(object):
    """A single model instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    # def __str__(self):
    #   s = ""
    #   s += "tokens: %s\n" % (" ".join(
    #       [tokenization.printable_text(x) for x in self.tokens]))
    #   s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    #   s += "is_random_next: %s\n" % self.is_random_next
    #   s += "masked_lm_positions: %s\n" % (" ".join(
    #       [str(x) for x in self.masked_lm_positions]))
    #   s += "masked_lm_labels: %s\n" % (" ".join(
    #       [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    #   s += "\n"
    #   return s

    # def __repr__(self):
    #   return self.__str__()

def write_instance_to_example_file(
        instances, tokenizer, max_seq_length, max_predictions_per_seq, output_file, no_nsp
):
    """Create TF example files from `TrainingInstance`s."""

    total_written = 0
    features = collections.OrderedDict()

    num_instances = len(instances)
    features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["masked_lm_positions"] = np.zeros(
        [num_instances, max_predictions_per_seq], dtype="int32"
    )
    features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    if not no_nsp:
        features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")

    for inst_index, instance in enumerate(tqdm(instances)):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        # next_sentence_label = 1 if instance.is_random_next else 0

        features["input_ids"][inst_index] = input_ids
        features["input_mask"][inst_index] = input_mask
        features["segment_ids"][inst_index] = segment_ids
        features["masked_lm_positions"][inst_index] = masked_lm_positions
        features["masked_lm_ids"][inst_index] = masked_lm_ids
        if not no_nsp:
            features["next_sentence_labels"][inst_index] = 1 if instance.is_random_next else 0

        total_written += 1

        # if inst_index < 20:
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in instance.tokens]))

        #   for feature_name in features.keys():
        #     feature = features[feature_name]
        #     values = []
        #     if feature.int64_list.value:
        #       values = feature.int64_list.value
        #     elif feature.float_list.value:
        #       values = feature.float_list.value
        #     tf.logging.info(
        #         "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    print("saving helper")
    # path = "/".join(output_file.split("/")[:-1])
    # if not os.path.exists(path):
    #     os.mkdir(path)
    f = h5py.File(output_file, "w")
    f.create_dataset("input_ids", data=features["input_ids"], dtype="i4", compression="gzip")
    f.create_dataset("input_mask", data=features["input_mask"], dtype="i1", compression="gzip")
    f.create_dataset("segment_ids", data=features["segment_ids"], dtype="i1", compression="gzip")
    f.create_dataset(
        "masked_lm_positions", data=features["masked_lm_positions"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "masked_lm_ids", data=features["masked_lm_ids"], dtype="i4", compression="gzip"
    )
    if not no_nsp:
        f.create_dataset(
            "next_sentence_labels",
            data=features["next_sentence_labels"],
            dtype="i1",
            compression="gzip",
        )
    f.flush()
    f.close()

def create_training_instances(
        input_files,
        tokenizer,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
        no_nsp,
        # information
):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        print("creating instance from {}".format(input_file))
        with open(input_file, "r") as reader:
            while True:
                line = convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents) # TODO: remove

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            if no_nsp:
                instances.extend(
                    create_instances_from_document_no_nsp(
                        all_documents,
                        document_index,
                        max_seq_length,
                        short_seq_prob,
                        masked_lm_prob,
                        max_predictions_per_seq,
                        vocab_words,
                        rng,
                        # information
                    )
                )
            else:
                instances.extend(
                    create_instances_from_document(
                        all_documents,
                        document_index,
                        max_seq_length,
                        short_seq_prob,
                        masked_lm_prob,
                        max_predictions_per_seq,
                        vocab_words,
                        rng,
                    )
                )

    rng.shuffle(instances) # TODO: remove
    return instances

def create_instances_from_document_no_nsp(
        all_documents,
        document_index,
        max_seq_length,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        # information
):
    """Creates `TrainingInstance`s for a single document."""
    """Generate single sentences (NO 2nd segment for NSP task)"""
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-model and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                tokens_a = []
                for j in range(len(current_chunk)):
                    tokens_a.extend(current_chunk[j])

                truncate_single_seq(tokens_a, max_num_tokens, rng)

                assert len(tokens_a) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                assert len(tokens) <= max_seq_length

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng,# information
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=False,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

def create_instances_from_document(
        all_documents,
        document_index,
        max_seq_length,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-model and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    # If picked random document is the same as the current document
                    if random_document_index == document_index:
                        is_random_next = False

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

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

def pmi_masking(old_indices, tokens):
    new_indices = []
    while old_indices:
        first_word_indices = old_indices.pop(random.randrange(len(old_indices)))
        new_indices.append(first_word_indices)
        first_word = create_word_from_indices(tokens, first_word_indices)
        if not old_indices or len(new_indices) >= 20:
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
            if current_pmi > best_pmi or second_word_indices is None:
                best_second_indices = second_word_indices
                best_pmi = current_pmi
        try:
            old_indices.remove(best_second_indices)
            new_indices.append(best_second_indices)
        except:
            return new_indices
    return new_indices

def flatten(l):
    return [item for sublist in l for item in sublist]

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):#, information):
    """Creates the predictions for the masked LM objective."""

    # information[0] is cooccurence, information[1] is single word probability

    cand_indexes = wwm(tokens) #
    #cand_indexes = pmi_masking(cand_indexes, tokens)#, information)
    cand_indexes = flatten(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        # if rng.random() < 0.8:
        masked_token = "[MASK]"
        # else:
        #     # 10% of the time, keep original
        #     if rng.random() < 0.5:
        #         masked_token = tokens[index]
        #     # 10% of the time, replace with random word
        #     else:
        #         masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def truncate_single_seq(tokens, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()

def create_pretraining_data(vocab_file, input_file, output_file, bert_model, no_nsp,
                            max_seq_length, dupe_factor, max_predictions_per_seq,
                            masked_lm_prob, short_seq_prob, do_lower_case, random_seed,
                            # information
                            ):

    tokenizer = BertTokenizer(vocab_file, do_lower_case=do_lower_case, max_len=512)

    input_files = []
    if os.path.isfile(input_file):
        input_files.append(input_file)
    elif os.path.isdir(input_file):
        input_files = [
            os.path.join(input_file, f)
            for f in os.listdir(input_file)
            if (os.path.isfile(os.path.join(input_file, f)) and f.endswith(".txt"))
        ]
    else:
        raise ValueError("{} is not a valid path".format(input_file))

    rng = random.Random(random_seed)
    instances = create_training_instances(
        input_files,
        tokenizer,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
        no_nsp,
        # information
    )

    output_file = output_file
    # try:
    write_instance_to_example_file(
        instances,
        tokenizer,
        max_seq_length,
        max_predictions_per_seq,
        output_file,
        no_nsp,
    )


def list_files_in_dir(dir, data_prefix=".txt"):
    dataset_files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and data_prefix in f
    ]
    return dataset_files

def open_with_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to shards dataset")
    parser.add_argument("-o", type=str, required=True, help="Output directory")
    parser.add_argument("--dup_factor", type=int, default=1, help="sentence duplication factor")
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--vocab_file", type=str, help="vocab file")
    parser.add_argument("--do_lower_case", type=int, default=1, help="lower case True = 1, False = 0")
    parser.add_argument("--masked_lm_prob", type=float, help="Specify the probability for masked lm", default=0.15)
    parser.add_argument("--max_seq_length", type=int, help="Specify the maximum sequence length", default=512)
    parser.add_argument("--model_name", type=str, required=True, help="Pre-trained models name (HF format).")
    parser.add_argument("--max_predictions_per_seq", type=int, help="Specify the maximum number of masked words per sequence", default=20)
    parser.add_argument("--n_processes", type=int, default=8, help="number of parallel processes")
    args = parser.parse_args()

    shard_files = list_files_in_dir(args.dir)
    new_shards_output = args.o
    os.makedirs(new_shards_output, exist_ok=True)
    logger.info("Creating new hdf5 files ...")

    def create_shard(f_path, shard_idx, set_group, args):#, information):
        create_pretraining_data(vocab_file=args.vocab_file if args.vocab_file is not None else "",
                                input_file=f_path,
                                output_file=os.path.join(new_shards_output, f"{set_group}_shard_{shard_idx}.hdf5"),
                                bert_model=args.model_name if args.model_name is not None else "",
                                no_nsp=True,
                                max_seq_length=args.max_seq_length,
                                dupe_factor=1,
                                max_predictions_per_seq=args.max_predictions_per_seq,
                                masked_lm_prob=args.masked_lm_prob,
                                short_seq_prob=0.1,
                                do_lower_case=args.do_lower_case,
                                random_seed=args.seed + shard_idx)#,
                                # information=information)

    cooc_path = "/mounts/work/kerem/Projects/pmi_masking/wiki_again/merge_final_from_merge_7/cooccurence.pickle"
    word_path = "/mounts/work/kerem/Projects/pmi_masking/wiki_again/5_merged_data/vocab.pickle"
    #manager = Manager()
    #from collections import defaultdict
    # freq = open_with_pickle(word_path)#defaultdict(int)#
    # cooc = open_with_pickle(cooc_path)#defaultdict(int)#
    freq = defaultdict(lambda: random.random()*10)
    cooc = defaultdict(lambda: random.random())
    #freq = open_with_pickle(word_path)#defaultdict(int)#
    # cooccurence = manager.dict(coocurence)
    # word_probs = manager.dict(word_probs)

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    counter = 0
    for dup_idx in range(args.dup_factor):
        for files in chunk(shard_files, args.n_processes):
            processes = []
            for f in files:
                p = Process(target=create_shard, args=(f, counter, "train", args))#, [cooccurence, word_probs]))
                counter += 1
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


"""
CIS Machine
python3 5_generate_samples.py \
--dir /mounts/data/proj/braasch/4_MergedShards \
-o /mounts/data/proj/braasch/5_MaskedSamples/ \
--dup_factor 10 \
--seed 40 \
--vocab_file ~/academic-budget-bert/dataset/data/bert_large_uncased_vocab.txt \
--masked_lm_prob 0.15 \
--max_seq_length 128 \
--model_name bert-large-uncased \
--max_predictions_per_seq 20 \
--n_processes 64
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

MY MACHINE
python3 5_generate_samples.py \
--dir ./data/Wikipedia/4_MergedShards \
-o ./data/Wikipedia/5_MaskedSamples/ \
--dup_factor 10 \
--seed 40 \
--vocab_file ./data/bert_large_uncased_vocab.txt \
--masked_lm_prob 0.15 \
--max_seq_length 128 \
--model_name bert-large-uncased \
--max_predictions_per_seq 20 \
--n_processes 1
"""
