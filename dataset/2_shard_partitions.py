import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the shards."
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output path for sharded partitions."
    )
    parser.add_argument(
        "--num_train_shards",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_test_shards",
        type=int,
        required=True
    )
    parser.add_argument(
        "--frac_test",
        type=float,
        required=True
    )
    args = parser.parse_args()
    return args

def shard_partitions():
    args = get_args()
    if not os.path.exists(args.o):
        os.mkdir(args.o)
    for nr in os.listdir(args.dir):
        input_name, output_name = f"{args.dir}{nr}", f"{args.o}{nr}"
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        command = f"python3 ../shard_data.py " \
                  f"--dir {input_name} " \
                  f"-o {output_name} " \
                  f"--num_train_shards {args.num_train_shards} " \
                  f"--num_test_shards {args.num_test_shards} " \
                  f"--frac_test {args.frac_test} "
        os.system(command)

shard_partitions()