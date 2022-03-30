# RUN : sh train.sh

python train.py \
--seed 42 \
--save_dir /opt/ml/code/results \
--wandb_path test-project \
--wandb_name test-project \
--train_path /opt/ml/dataset/train/train.csv \
--tokenize_option PUN \
--fold 5 \
--model klue/roberta-large \
--loss LB \
--epochs 1 \
--lr 5e-5 \
--batch 16 \
--gradient_accum 2 \
--batch_valid 16 \
--warmup 0.1 \
--eval_steps 500 \
--save_steps 500 \
--logging_steps 100 \
--weight_decay 0.01 