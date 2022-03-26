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

import argparse
import logging
import os
import subprocess
from multiprocessing import Manager, Process
from helper.create_pretraining_data import create_pretraining_data
import pickle5 as pickle
from collections import defaultdict


logger = logging.getLogger()

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

    def create_shard(f_path, shard_idx, set_group, args, information):
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
                                random_seed=args.seed + shard_idx,
                                information=information)

    cooc_path = "/mounts/work/kerem/Projects/pmi_masking/wiki_again/merge_final_from_merge_7/cooccurence.pickle"
    word_path = "/mounts/work/kerem/Projects/pmi_masking/wiki_again/merge_final_from_merge_7/vocab.pickle"
    manager = Manager()
    coocurence = defaultdict(int)#open_with_pickle(cooc_path)
    word_probs = defaultdict(int)#open_with_pickle(word_path)
    cooccurence = manager.dict(coocurence)
    word_probs = manager.dict(word_probs)

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    counter = 0
    for dup_idx in range(args.dup_factor):
        for files in chunk(shard_files, args.n_processes):
            processes = []
            for f in files:
                p = Process(target=create_shard, args=(f, counter, "train", args, [cooccurence, word_probs]))
                counter += 1
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


"""
python3 5_generate_samples.py \
--dir ./data/Wikipedia/4_MergedShards/ \
-o ./data/Wikipedia/5_MaskedSamples/ \
--dup_factor 10 \
--seed 40 \
--vocab_file ./data/bert_large_uncased_vocab.txt \
--masked_lm_prob 0.15 \
--max_seq_length 128 \
--model_name bert-large-uncased \
--max_predictions_per_seq 20 \
--n_processes 30
"""