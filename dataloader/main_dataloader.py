import sys
sys.path.append('..')
import os
import pandas as pd
from typing import List, Dict, Tuple
from dataset.main_dataset import *
from preprocess.main_preprocess import *
from constants import * 
from pickled_data.main_pickle import *
from augmentation.generate import *


def load_data(dataset_dir:str, train=True, generate=True)->pd.DataFrame:
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    if train and os.path.isfile(PKL_TRAIN_PATH):
        return load_preprocessed_data(PKL_TRAIN_PATH)

    if not train and os.path.isfile(PKL_TEST_PATH):
        return load_preprocessed_data(PKL_TEST_PATH)

    if generate:
        existed_dataset = pd.read_csv(dataset_dir)
        generated_dataset = load_generate_data()
        pd_dataset = pd.concat([existed_dataset, generated_dataset], ignore_index=True)
    else:
        pd_dataset = pd.read_csv(dataset_dir)

    dataset = preprocessing_dataset(pd_dataset, train)
    
    return dataset

def load_test_dataset(dataset_dir:str, tokenizer)->Tuple[List[int], Dict, List[int]]:
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, False)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label
