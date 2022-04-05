import pandas as pd
import torch
from typing import List, Dict, Tuple

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class. """
    def __init__(self, pair_dataset:Dict, labels:List[int]):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) 

def tokenized_dataset(df: pd.DataFrame, tokenizer) -> Dict:
    """
    Add Query로 데이터에 추가하여 tokenizer에 따라 sentence를 tokenizing 합니다.
    이 때 subject_entity와 object_entity 순서를 고려하여 Query(question)을 생성합니다. 

    추가 : @*[SUBT]PER[/SUBT]*[SUB]이순신[/SUB]@과 #^[OBJT]POH[/OBJT]^[OBJ]무신[/OBJ]#의 관계를 나타내는 문장  
    """
    new_question = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        subject_entity, subject_type =(
            row['subject_entity'],
            row['subject_type'],
        )

        object_entity, object_type =(
            row['object_entity'],
            row['object_type'],
        )
        
        question = f'@*[SUBT]{subject_type}[/SUBT]*[SUB]{subject_entity}[/SUB]@와 \
            #^[OBJT]{object_type}[/OBJT]^[OBJ]{object_entity}[/OBJ]#의 관계를 나타내는 문장'
        new_question.append(question)

    tokens = ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']
    tokenizer.add_tokens(tokens)
    tokenizer.add_special_tokens({"additional_special_tokens":\
    ['[MASK]', '[SUBT]', '[/SUBT]', '[SUB]', '[/SUB]', '[OBJT]', '[/OBJT]', '[OBJ]', '[/OBJ]']})
        
    tokenized_sentences = tokenizer(
        new_question,
        list(df['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
        
    return tokenized_sentences
