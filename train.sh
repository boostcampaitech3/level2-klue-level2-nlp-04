# RUN : sh train.sh

python main_train.py \
--seed 42 \
--wandb_path test-project \
--wandb_name refactor-test-2 \
--split_ratio 0.2 \
--fold 5 \
--model klue/roberta-large \
--loss LB \
--epochs 1 \
--lr 5e-5 \
--batch 16 \
--batch_valid 16 \
--warmup 0.1 \
--eval_steps 500 \
--save_steps 500 \
--logging_steps 100 \
--weight_decay 0.01 