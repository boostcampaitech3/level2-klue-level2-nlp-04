import pickle
import pandas as pd
import torch
from preprocess.inbeom_preprocess import *


def load_data_v2(dataset_dir: str) -> pd.DataFrame:
    """ 
    csv 파일을 경로에 맡게 불러 옵니다.
     """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset_v2(pd_dataset)
  
    return dataset

