import os
import random
import time

command = lambda seed, i: \
    f"""deepspeed run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule time \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 48.0 \
  --early_exit_time_marker 12.0 \
  --dataset_path /home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Wikipedia/MaskedSamples \
  --output_dir /home/marcelbraasch/PycharmProjects/academic-budget-bert/dataset/data/Model-{i} \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name Dropped \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 120 \
  --seed {seed} \
  --fp16"""

# os.system("cd ~/PycharmProjects/academic-budget-bert/")
# os.system("source venv/bin/activate")
i = 0
finish = False
while not finish:
    start = time.time()
    seed = random.randrange(1,100)
    os.system(command(seed, i))
    end = time.time()
    i += 1
    if end - start > 18000: # 5 hours
        finish = True

