import os
import pandas as pd 
import torch
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
  AutoTokenizer,
  AutoConfig,
  AutoModelForSequenceClassification,
  Trainer,
  TrainingArguments,
  EarlyStoppingCallback
  )
from transformers.utils import logging
from load_data import *
import wandb
import argparse
from utilities.main_utilities import *

def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data(args.train_path)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    X_train, X_val = train_test_split(RE_train_dataset, test_size=0.2, random_state=42)

    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Training Start")
    print("="*100)
    print(f"DEVICE : {device}")
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      output_dir=args.save_dir,          # output directory
      save_total_limit=5,              # number of total save model.
      save_steps=args.save_steps,                 # model saving step.
      num_train_epochs=args.epochs,              # total number of training epochs
      learning_rate=args.lr,               # learning_rate
      per_device_train_batch_size=args.batch,  # batch size per device during training
      per_device_eval_batch_size=args.batch_valid,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=args.warmup,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
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
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=X_train,         # training dataset
      eval_dataset=X_val,             # evaluation dataset
      compute_metrics=compute_metrics,         # define metrics function
      callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model')


def main():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default = './results', 
                        help='model save dir path (default : ./results)')
    parser.add_argument('--wandb_path', type= str, default= 'test-project',
                        help='wandb graph, save_dir basic path (default: test-project') 
    parser.add_argument('--train_path', type= str, default= '/opt/ml/dataset/train/train.csv',
                        help='train csv path (default: /opt/ml/dataset/train/train.csv')
    parser.add_argument('--tokenize_option', type=str, default='PUN',
                        help='token option ex) SUB, PUN')    
    parser.add_argument('--fold', type=int, default=5,
                        help='fold (default: 5)')
    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--loss', type=str, default= 'LB',
                        help='LB: LabelSmoothing, CE: CrossEntropy')
    parser.add_argument('--wandb_name', type=str, default= 'test',
                        help='wandb name (default: test)')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--batch', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--gradient_accum', type=int, default=2,
                        help='gradient accumulation (default: 2)')
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
                        help='metric_for_best_model (default: f1')
    
    args= parser.parse_args()
    wandb.init(name=args.wandb_name, project=args.wandb_path, entity="boostcamp_nlp_04", config = vars(args),)

    logging.set_verbosity_warning()
    logger = logging.get_logger()
    logger.warning("\n")
    
    train(args)

if __name__ == '__main__':
    main()
