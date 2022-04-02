import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_dir", type=str, required=True, help="Path to the directory of finetuning checkpoints.")
    parser.add_argument("-o", type=str, required=True, help="Path to the directory of finetuning results")
    args = parser.parse_args()
    tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    finetuning_ckpts = next(os.walk(args.ckpts_dir))[2]
    for ckpt in finetuning_ckpts:
        for i, task in enumerate(tasks):
            command = f"""python run_glue.py 
                          --model_name_or_path {args.ckpts_dir}/{ckpt} 
                          --task_name {task} 
                          --max_seq_length 128
                          --output_dir {args.ckpts_dir}/{task}/{ckpt} 
                          --overwrite_output_dir 
                          --do_train 
                          --do_eval 
                          --do_test 
                          --evaluation_strategy steps 
                          --per_device_train_batch_size 32 
                          --gradient_accumulation_steps 1 
                          --per_device_eval_batch_size 32 
                          --learning_rate 5e-5 
                          --weight_decay 0.01 
                          --eval_steps 50 
                          --evaluation_strategy steps 
                          --max_grad_norm 1.0 
                          --num_train_epochs 5 
                          --lr_scheduler_type polynomial 
                          --warmup_steps 50"""
            os.system(command)

if __name__ == "__main__":
    main()