import sys
sys.path.append("..")
from constants import *
import pandas as pd
import re
from typing import List
from pickled_data.main_pickle import *


def remove_duplicate_row(df:pd.DataFrame) -> pd.DataFrame:
    """
    mislabel된 row를 제거한 clean_train.csv를 생성합니다
    """
    i_clean_df =df.drop([6749, 8364, 11511, 277, 10202, 4212]) # mislabeled row
    i_clean_df = i_clean_df.drop_duplicates(['sentence','subject_entity','object_entity', 'label'], keep='first', inplace=False, ignore_index = True) # 중복 row 제거
    clean_id = i_clean_df['id']
    clean_df = df[df['id'].isin(clean_id)]

    return clean_df

def remove_sidespace(texts: List) -> List:
    """
    문장의 양 옆 반 공간을 제거합니다.
    """
    preprocessed_text = []
    for text in texts:
        text = text.strip()
        preprocessed_text.append(text)

    return preprocessed_text

def remove_repeated_spacing(texts: List) -> List:
    """
    두 개 이상의 연속된 공백을 하나로 치환합니다.
    ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        preprocessed_text.append(text)

    return preprocessed_text

def remove_special_char(texts: List) -> List:
    """
    불필요한 특수 기호를 제거합니다.
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"[À-ÿ]+", "", text)
        text = re.sub(r"[\u0600-\u06FF]+", "", text)
        text = re.sub(r"[\u00C0-\u02B0]+", "", text)
        text = re.sub(r"[ß↔Ⓐب€☎☏±∞『』▲㈜]+", "", text)
        preprocessed_text.append(text)
    
    return preprocessed_text

def clean_punc(texts: List) -> List: 
    """
    특수 기호를 일반화를 진행합니다.
    추가로 일반화를 진행하고 싶은 문자의 경우, extra_mapping에 추가합니다.
    """
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", \
                    "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', \
                    "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', \
                    'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

    extra_mapping = {"《": "(",
                     "》": ")", 
                     "<": "(", 
                     ">": ")", 
                     "[": "(", 
                     "]": ")"}
                     
    punct_mapping.update(extra_mapping)

    preprocessed_text = []
    for text in texts:
        for p in punct_mapping:
            text = text.replace(p, punct_mapping[p])
        text = text.strip()
        preprocessed_text.append(text)  

    return preprocessed_text

def preprocessing_dataset(dataset: pd.DataFrame, generate_option:int, train=True) -> pd.DataFrame:
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    
    subject_entity (변겅 전) -> subject_entity, subject_type, subject_idx (변경 후)
    {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} (변경 전) -> "비틀즈", "(24, 26)", "ORG" (변경 후)
    """
    subject_entity = []
    subject_type = []
    subject_idx = []

    object_entity = []
    object_type = []
    object_idx =[]

    for sub, obj in zip(dataset['subject_entity'], dataset['object_entity']):

        sub_entity = eval(sub)['word']
        sub_type = eval(sub)['type']
        sub_idx = (int(eval(sub)['start_idx']), int(eval(sub)['end_idx']))

        obj_entity = eval(obj)['word']
        obj_type = eval(obj)['type']
        obj_idx = (int(eval(obj)['start_idx']), int(eval(obj)['end_idx']))

        subject_entity.append(sub_entity)
        subject_type.append(sub_type)
        subject_idx.append(sub_idx)

        object_entity.append(obj_entity)
        object_type.append(obj_type)
        object_idx.append(obj_idx)

    dataset = pd.DataFrame({'id':dataset['id'], 
                                'sentence':dataset['sentence'],
                                'subject_entity':subject_entity, 'subject_type':subject_type, 'subject_idx':subject_idx,
                                'object_entity':object_entity, 'object_type':object_type, 'object_idx':object_idx,
                                'label':dataset['label'],})                    


    if train: 
        dataset = dataset.pipe(remove_duplicate_row)\
                         .pipe(typed_entity_marker_with_punctuation)
    else:
        dataset = dataset.pipe(typed_entity_marker_with_punctuation)

    feature = ['sentence', 'subject_entity', 'object_entity']

    for feat in feature:
        dataset[feat] = remove_repeated_spacing(remove_special_char(clean_punc(dataset[feat])))


    if train:
        save_preprocessed_data(f'{PKL_TRAIN_PATH}_{generate_option}.pkl', dataset)
    else:
        save_preprocessed_data(f'{PKL_TEST_PATH}_{generate_option}.pkl', dataset)

    return dataset

def typed_entity_marker_with_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Typed Entity Marker with Punctuation(=sentence) 방식으로 기존의 sentence를 변경합니다. 
    이 때 subject_entity와 object_entity 순서를 고려하여 sentence를 변경합니다. 

    변경 전 : 이순신은 조신 시대 중기의 무신이다.
    변경 후 : @*PER*이순신@은 조선 시대 중기의 #^POH^무신#이다.     
    """
    new_sentence = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        sentence = row['sentence']
        subject_entity, subject_type, subject_idx=(
            row['subject_entity'],
            row['subject_type'],
            row['subject_idx'],
        )
        object_entity, object_type, object_idx=(
            row['object_entity'],
            row['object_type'],
            row['object_idx'],
        )

        if subject_idx[0] < object_idx[0]:
            sentence = (
                sentence[:subject_idx[0]]
                + '@*'
                + subject_type 
                + '*' 
                + subject_entity
                + '@' 
                + sentence[subject_idx[1]+1:object_idx[0]]
                + '#^' 
                + object_type
                + '^' 
                + object_entity 
                + '#' 
                + sentence[object_idx[1]+1:]
            )
        else:
            sentence = (
                sentence[:object_idx[0]]
                + '#^'
                + object_type 
                + '^' 
                + object_entity
                + '#' 
                + sentence[object_idx[1]+1:subject_idx[0]]
                + '@*' 
                + subject_type
                + '*'
                + subject_entity 
                + '@'
                + sentence[subject_idx[1]+1:]
            )
        
        new_sentence.append(sentence)

    df['sentence'] = new_sentence

    return df