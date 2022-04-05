# RUN : sh train.sh

python main_train.py \
--seed 42 \
--wandb_path test-project \
--wandb_name full-augmentation-10p-focal \
--split_ratio 0.2 \
--fold 5 \
--model klue/roberta-large \
--loss focal \
--epochs 10 \
--lr 5e-6 \
--batch 16 \
--batch_valid 16 \
--warmup 0.1 \
--eval_steps 500 \
--save_steps 500 \
--logging_steps 1000 \
--augmentation True \
--weight_decay 0.02