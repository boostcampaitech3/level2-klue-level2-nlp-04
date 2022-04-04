import sys
sys.path.append('..')
import pandas as pd
from typing import List, Dict, Tuple
from preprocess.jsh_preprocess import *

def load_data(dataset_dir:str, train=True)->pd.DataFrame:
    """ csv 파일을 경로에 맡게 불러 옵니다. """
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
    tokenized_test, _ = tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label

def tokenized_dataset(dataset:pd.DataFrame, tokenizer)->Dict:
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    tokens= ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']

    added_token_num = tokenizer.add_tokens(tokens)
    added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":\
                    ["[SUB]", "[/SUB]", "[OBJ]", "[/OBJ]", "[SUBT]", "[/SUBT]", "[OBJT]", "[/OBJT]"]})

    concat_entity = []
    for sub, obj, sub_t, obj_t in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
        temp = f'@*[SUBT]{sub_t}[/SUBT]*[SUB]{sub}[/SUB]@과 #^[OBJT]{obj_t}[/OBJT]^[OBJ]{obj}[/OBJ]#의 관계를 나타내는 문장'
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
        
    return tokenized_sentences, added_token_num