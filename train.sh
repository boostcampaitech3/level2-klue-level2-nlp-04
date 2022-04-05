# RUN : sh train.sh

python main_train.py \
--seed 42 \
--wandb_path test-project \
--wandb_name refactor-test \
--split_ratio 0.2 \
--fold 5 \
--model klue/roberta-large \
--loss focal \
--epochs 1 \
--lr 5e-6 \
--batch 32 \
--batch_valid 32 \
--warmup 0.1 \
--eval_steps 500 \
--save_steps 500 \
--logging_steps 1000 \
--augmentation False \
--generate_option 0 \
--weight_decay 0.02