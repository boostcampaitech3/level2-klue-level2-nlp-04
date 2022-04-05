from genericpath import exists
from operator import ge
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


def load_data(dataset_dir:str, generate_option:int,train=True)->pd.DataFrame:
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    if train and os.path.isfile(f'{PKL_TRAIN_PATH}_{generate_option}.pkl'):
        return load_preprocessed_data(f'{PKL_TRAIN_PATH}_{generate_option}.pkl')

    if not train and os.path.isfile(f'{PKL_TEST_PATH}_{generate_option}.pkl'):
        return load_preprocessed_data(f'{PKL_TEST_PATH}_{generate_option}.pkl')

    if generate_option == ONLY_ORIGINAL: #원본 데이터만 쓰는 경우
        pd_dataset = pd.read_csv(dataset_dir)

    elif generate_option == ONLY_GENERATED:
        pd_dataset = load_generate_data() 

    else:
        existed_dataset = pd.read_csv(dataset_dir)
        generated_dataset = load_generate_data()
        pd_dataset = pd.concat([existed_dataset, generated_dataset], ignore_index=True)
    

    dataset = preprocessing_dataset(pd_dataset, generate_option, train)
    
    return dataset

def load_test_dataset(dataset_dir:str, tokenizer)->Tuple[List[int], Dict, List[int]]:
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, ONLY_ORIGINAL, False)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label
