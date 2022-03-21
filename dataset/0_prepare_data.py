import argparse
import os
import shutil
from time import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_train_shards",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_test_shards",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--frac_test",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--dup_factor",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
    )
    # parser.add_argument(
    #     "--do_lower_case",
    #     type=bool,
    #     required=True,
    # )
    parser.add_argument(
        "--masked_lm_prob",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    return args

def main():

    start = time()
    args = get_args()
    working_dir = args.working_dir[:-1] if args.working_dir[-1] == "/" else args.working_dir

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # 2. Create n partitions of raw huggingface wikipedia (we need this in case of limited RAM)
    command = f"python3 1_process_raw_to_shardable.py " \
              f"--dir {working_dir}/1_Dataset/ " \
              f"-o {working_dir}/2_Split/ " \
              f"--splits {args.splits} "
    os.system(command)
    """
    python3 1_process_raw_to_shardable.py \
    --dir ./data/1_Dataset/ \
    -o ./data/2_Split/  \
    --splits 4
    """
    # shutil.rmtree(f"{working_dir}/1_Dropped/")


    # 3. Shard partions
    command = f"python3 2_shard_partitions.py " \
              f"--dir {working_dir}/2_Split/ " \
              f"-o {working_dir}/3_Shards/ " \
              f"--frac_test {args.frac_test} " \
              f"--num_train_shards {args.num_train_shards} " \
              f"--num_test_shards {args.num_test_shards} "
    os.system(command)
    """
    python3 2_shard_partitions.py \
    --dir ./data/2_Split/ \
    -o /mounts/data/proj/braasch/3_Shards \
    --frac_test 0.1 \
    --num_train_shards 4096 \
    --num_test_shards 2048
    """
    #shutil.rmtree(f"{working_dir}/2_Split/")


    # 6. Merge shards to create lesser files
    command = f"python3 4_merge_shards.py " \
              f"--data {working_dir}/3_Shards/ " \
              f"--output_dir {working_dir}/4_MergedShards/ " \
              f"--ratio {2*2} "
    os.system(command)
    """
    python3 4_merge_shards.py \
    --data /mounts/data/proj/braasch/3_shards \
    --output_dir /mounts/data/proj/braasch/4_MergedShards/ \
    --ratio 6
    """
    #shutil.rmtree(f"{working_dir}/3_Shards/")

    #     # 7. Generate Samples
    command = f"python 5_generate_samples.py " \
              f"--dir {working_dir}/4_MergedShards/ " \
              f"-o {working_dir}/5_MaskedSamplestest/ " \
              f"--dup_factor {args.dup_factor} " \
              f"--seed {args.seed} " \
              f"--vocab_file {args.vocab_file} " \
              f"--masked_lm_prob {args.masked_lm_prob} " \
              f"--max_seq_length {args.max_seq_length} " \
              f"--model_name {args.model_name} " \
              f"--max_predictions_per_seq {args.max_predictions_per_seq} " \
              f"--n_processes {args.num_workers} "
    """
    python3 5_generate_samples.py \
    --dir /mounts/data/proj/braasch/4_MergedShards/ \
    -o /mounts/data/proj/braasch/5_MaskedSamples/ \
    --dup_factor 10 \
    --seed 40 \
    --vocab_file ~/academic-budget-bert/dataset/data/bert_large_uncased_vocab.txt \
    --masked_lm_prob 0.15 \
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 64
    """

    os.system(command)
    #shutil.rmtree(f"{working_dir}/4_MergedShards/")
    

    end = time()


    print("\n" * 5)
    print(f"Finishing processing data within {round((((end-start)/60)/60), 2)} hours.")

main()