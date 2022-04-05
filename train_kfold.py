import os
import pandas as pd 
import torch
import sklearn
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from transformers import (
  AutoTokenizer,
  AutoConfig,
  AutoModelForSequenceClassification,
  Trainer,
  TrainingArguments,
  EarlyStoppingCallback
  )
from transformers.utils import logging
import wandb
import argparse
from utilities.main_utilities import *
from utilities.criterion.loss import *
from dataloader.main_dataloader import *
from dataset.main_dataset import *
from preprocess.main_preprocess import *
from constants import *
from augmentation.main_augmentation import *

class CustomTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        if self.loss_name == 'CE':
          loss_fct = nn.CrossEntropyLoss
        elif self.loss_name == 'LB':
          loss_fct = LabelSmoothingLoss()
        elif self.loss_name == 'focal':
          loss_fct = FocalLoss()
        elif self.loss_name == 'f1':
          loss_fct = F1Loss()
          
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def fold_selection(args):
    model_select = None

    if args.kfold == 'KFold':
        model_select = KFold(n_splits=args.fold)
    elif args.kfold == 'StratifiedKFold':
        model_select = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        
    return model_select
  
def train(args):
    # load model and tokenizer
    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_data(TRAIN_DIR)
    label = dataset['label'].values
    
    kfold = fold_selection(args)
    for K ,(train_index, dev_index) in enumerate(kfold.split(dataset, label)):
        wandb.init(name=f'{args.wandb_name}_{K}', project=args.wandb_path, entity=WANDB_ENT, config = vars(args),)

        # load dataset
        train_dataset = dataset.iloc[train_index]
        dev_dataset = dataset.iloc[dev_index]

        train_label = label_to_num(train_dataset['label'].values)
        dev_label = label_to_num(dev_dataset['label'].values)

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        if args.augmentation:
          tokenized_train = main_augmentation(tokenized_train)
          tokenized_dev =  main_augmentation(tokenized_dev)
        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print(f"Training Start({K})")
        print("="*100)
        print(f"DEVICE : {device}")
        # setting model hyperparameter
        model_config =  AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        model.resize_token_embeddings(tokenizer.vocab_size + args.add_token)
        model.to(device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir=f'{SAVE_DIR}/{K}',          # output directory
            save_total_limit=5,              # number of total save model.
            save_steps=args.save_steps,                 # model saving step.
            num_train_epochs=args.epochs,              # total number of training epochs
            learning_rate=args.lr,               # learning_rate
            per_device_train_batch_size=args.batch,  # batch size per device during training
            per_device_eval_batch_size=args.batch_valid,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=args.warmup,               # strength of weight decay
            logging_dir=LOG_DIR,            # directory for storing logs
            logging_steps=args.logging_steps,              # log saving step.
            evaluation_strategy='steps', # evaluation strategy to adopt during training
                                        # `no`: No evaluation during training.
                                        # `steps`: Evaluate every `eval_steps`.
                                        # `epoch`: Evaluate every end of epoch.
            eval_steps = args.eval_steps,            # evaluation step.
            load_best_model_at_end = True,
            metric_for_best_model = args.metric_for_best_model,
            report_to='wandb' 
        )

        trainer = CustomTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset=RE_dev_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,         # define metrics function
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
            loss_name = args.loss
        )
        wandb.finish()
        # train model
        trainer.train()
        path = os.path.join(BEST_MODEL_DIR, f'{args.wandb_name}{K}')
        model.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--wandb_path', type= str, default= 'test-project',
                        help='wandb graph, save_dir basic path (default: test-project') 
    parser.add_argument('--fold', type=int, default=5,
                        help='fold (default: 5)')
    parser.add_argument('--kfold', type=str, default='StratifiedKFold',
                        help='StratifiedKFold(default) / KFold')                    
    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--loss', type=str, default= 'LB',
                        help='LB: LabelSmoothing, CE: CrossEntropy, focal: Focal, f1:F1loss')
    parser.add_argument('--wandb_name', type=str, default= 'test',
                        help='wandb name (default: test)')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--batch', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--batch_valid', type=int, default=16,
                        help='input batch size for validing (default: 16)')
    parser.add_argument('--warmup', type=float, default=0.1,
                        help='warmup_ratio (default: 0.1)')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='eval_steps (default: 500)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='save_steps (default: 500)')
    parser.add_argument('--logging_steps', type=int,
                        default=100, help='logging_steps (default: 100)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='weight_decay (default: 0.01)')
    parser.add_argument('--metric_for_best_model', type=str, default='f1',
                        help='metric_for_best_model (default: f1)')
    parser.add_argument('--add_token', type=int, default=14,
                        help='add token count (default: 14)')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                        help='Test Val split ratio (default : 0.2)')
    parser.add_argument('--augmentation', type=bool, default=True,
                        help='Apply Random Masking/Delteing (default=False)')
    
    args= parser.parse_args()
    
    logging.set_verbosity_warning()
    logger = logging.get_logger()
    logger.warning("\n")
    
    train(args)

if __name__ == '__main__':
    main()