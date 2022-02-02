import argparse
import os
import shutil
from time import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the shards."
    )
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
    parser.add_argument(
        "--do_lower_case",
        type=int,
        required=True,
    )
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

    # 1. Create new dataset from LexicalScores annotated dataset
    command = f"python3 create_new_dataset.py " \
              f"--num_workers {args.num_workers} " \
              f"--dir {args.dataset_path} " \
              f"-o {working_dir}/1_Dropped/ "
    os.system(command)


    # 2. Create n partitions of raw huggingface wikipedia (we need this in case of limited RAM)
    command = f"python3 process_raw_to_shardable.py " \
              f"--dir {working_dir}/1_Dropped/ " \
              f"-o {working_dir}/2_Split/ " \
              f"--splits {args.splits} "
    os.system(command)
    # shutil.rmtree(f"{working_dir}/1_Dropped/")


    # 3. Shard partions
    command = f"python3 shard_partitions.py " \
              f"--dir {working_dir}/2_Split/ " \
              f"-o {working_dir}/3_Shards/ " \
              f"--frac_test {args.frac_test} " \
              f"--num_train_shards {args.num_train_shards} " \
              f"--num_test_shards {args.num_test_shards} "
    os.system(command)
    shutil.rmtree(f"{working_dir}/2_Split/")

    # 3. Rename shard, move to one folder and delete subfolders
    command = f"python3 rename_files_in_shard.py " \
              f"--dir {working_dir}/3_Shards/ "
    os.system(command)

    # 6. Merge shards to create lesser files
    command = f"python3 ../merge_shards.py " \
              f"--data {working_dir}/3_Shards/ " \
              f"--output_dir {working_dir}/4_MergedShards/ " \
              f"--ratio {args.splits*2} "
    os.system(command)
    shutil.rmtree(f"{working_dir}/3_Shards/")

    # 7. Generate Samples
    command = f"python ../generate_samples.py " \
              f"--dir {working_dir}/4_MergedShards/ " \
              f"-o {working_dir}/5_MaskedSamples/ " \
              f"--dup_factor {args.dup_factor} " \
              f"--seed {args.seed} " \
              f"--vocab_file {args.vocab_file} " \
              f"--do_lower_case {args.do_lower_case}  " \
              f"--masked_lm_prob {args.masked_lm_prob} " \
              f"--max_seq_length {args.max_seq_length} " \
              f"--model_name {args.model_name} " \
              f"--max_predictions_per_seq {args.max_predictions_per_seq} " \
              f"--n_processes {args.num_workers} "

    os.system(command)
    shutil.rmtree(f"{working_dir}/4_MergedShards/")
    end = time()

    print("\n" * 5)
    print(f"Finishing processing data within {round((((end-start)/60)/60), 2)} hours.")

main()