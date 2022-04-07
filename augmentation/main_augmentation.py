from xml.dom import INDEX_SIZE_ERR
import torch
import numpy as np
from typing import List

tokenized_to_protect = [0,1,2,3,7,14,36,65,21639,32000,32001,32002,32003,32004,32005,32006,\
                       32007,32008,32009,32010,32011,32012]
#[cls, pad, sep, unk, #, *, @, ^, per, loc, poh, dat, noh, org,\
#[SUBT], [/SUBT], [SUB], [/SUB], [OBJT], [/OBJT], [OBJ], [/OBJ]]

def random_masking_or_delete(tokenize_train, indices:List[List[int]]):
    modify_train = tokenize_train

    for idx, index_list in enumerate(indices):
        if len(index_list) < 3:
            continue

        modified_id = tokenize_train['input_ids'][idx].tolist()
        rand = np.random.random()

        for i in index_list:
            if modified_id[i] not in tokenized_to_protect:
                if rand < 0.4:
                    modified_id[i] = 4
                elif rand >= 0.4 and rand <= 0.8:
                    modified_id.pop(i)
                    modified_id.append(2)

        modify_train['input_ids'][idx] = torch.tensor(modified_id)

    return modify_train 

def main_augmentation(tokenized_train, p=0.15):
    rand = np.random.random()

    # valid_indices 는 p 확률만큼 input_ids에서 랜덤으로 선택된 토큰의 인덱스를 담는 리스트    
    valid_indices = [[0] for _ in range(len(tokenized_train.input_ids))]


    # input_ids 에서 2는 [SEP] 토큰임.
    # 그러면 하나의 input에는 2개의 [SEP] 토큰이 있는데 그 중 첫번째가 Query 문과 Sentence 를 구분짓는 것임.
    # 아래 for 문은 valid_indices에 sentence 의 토큰들의 index들을 다 담는 과정
    for idx, input_ids in enumerate(tokenized_train.input_ids):
        sent_idx = np.where(input_ids == 2)[0].tolist()[0]+1

        # PAD 전까지의 토큰의 인덱스
        valid_indices[idx] = list(map(lambda x:x+sent_idx, np.where(input_ids[sent_idx:]>1)[0].tolist())) 


    # 아래 for 문은 담긴 인덱스들에서 랜덤하게 p의 확률만큼 뽑는 작업
    for idx, indices in enumerate(valid_indices):
        valid_indices[idx] = list(np.random.randint(low=min(indices), high=max(indices), size=int(len(indices)*p)))
    
    # 50%의 확률로 선택된 인덱스들의 토큰들은 [MASK] 로 치환됨
    # 20%의 확률로 선택된 인덱스들의 토큰들은 삭제됨 
    # 10%의 확률로 원본을 리턴함

    return random_masking_or_delete(tokenized_train, valid_indices)
