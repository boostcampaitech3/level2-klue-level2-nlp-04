import pandas as pd
import sys
sys.path.append("..")
from constants import *
import re
import pickle
import os

def remove_sidespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    문장의 양 옆 반 공간을 제거합니다.
    """
    df['sentence'] = df['sentence'].apply(lambda x : x.strip())

    return df

def remove_repeated_spacing(df: pd.DataFrame) -> pd.DataFrame:
    """
    두 개 이상의 연속된 공백을 하나로 치환합니다.
    ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
    """
    df['sentence'] = df['sentence'].apply(lambda sentence : re.sub(r"\s+", " ", sentence).strip())
    
    return df

def remove_special_char(df: pd.DataFrame) -> pd.DataFrame:
    """
    불필요한 특수 기호를 제거합니다.
    """
    df['sentence'] = df['sentence'].apply(lambda sentence : re.sub('[=+#/\?:^$@*\※~&%ㆍ!『』☎▲\|\…·]', '', sentence).strip())
    
    return df

def clean_punc(df: pd.DataFrame) -> pd.DataFrame: 
    """
    특수 기호를 일반화를 진행합니다.
    추가로 일반화를 진행하고 싶은 문자의 경우, extra_mapping에 추가합니다.
    """
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", \
                    "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', \
                    "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', \
                    'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

    extra_mapping = {"《": "(", "》": ")", "<": "(", ">": ")", "[": "(", "]": ")"}

    punct_mapping.update(extra_mapping)
    preprocessed_text = []
    for text in df['sentence'].values.tolist():
        for p in punct_mapping:
            text = text.replace(p, punct_mapping[p])
        text = text.strip()
        preprocessed_text.append(text)

    df['sentence'] = preprocessed_text

    return df

def save_preprocessed_data(save_path:str, data:pd.DataFrame):
    file = open(save_path, "wb")
    pickle.dump(data, file)
    file.close()
    return None

def load_preprocessed_data(save_path:str):
    f = open(save_path, "rb")
    data = pickle.load(f)
    f.close()
    return data

def preprocessing_dataset(dataset:pd.DataFrame, train=True)->pd.DataFrame:
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

    if train:
        if os.path.isfile(PKL_TRAIN_PATH):
            return load_preprocessed_data(PKL_TRAIN_PATH)

    dataset =dataset.pipe(remove_sidespace)\
                    .pipe(remove_repeated_spacing)\
                    .pipe(remove_special_char)\
                    .pipe(clean_punc)

    subject_entity = []
    object_entity = []
    subject_type = []
    object_type = []
    
    for sub, obj in zip(dataset['subject_entity'], dataset['object_entity']):
        s_e = sub[1:-1].split(',')[0].split(':')[1].replace("\'", "").strip()
        o_e = obj[1:-1].split(',')[0].split(':')[1].replace("\'", "").strip()
        s_t = sub[1:-1].split(',')[-1].split(':')[-1].replace("\'", "").strip()
        o_t = obj[1:-1].split(',')[-1].split(':')[-1].replace("\'", "").strip() 

        subject_entity.append(s_e)
        object_entity.append(o_e)
        subject_type.append(s_t)
        object_type.append(o_t)

    out_dataset = pd.DataFrame({'id':dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity':subject_entity,
                                'object_entity':object_entity,
                                'subject_type':subject_type,
                                'object_type':object_type,
                                'label':dataset['label']})

    if train:
        save_preprocessed_data(PKL_TRAIN_PATH, out_dataset) 
    else:
        save_preprocessed_data(PKL_TEST_PATH, out_dataset)

    return out_dataset