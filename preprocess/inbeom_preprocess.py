import pandas as pd
import re

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


def preprocessing_dataset_v2(dataset: pd.DataFrame) -> pd.DataFrame:
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
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
        sub_idx = (eval(sub)['start_idx'], eval(sub)['end_idx'])

        obj_entity = eval(obj)['word']
        obj_type = eval(obj)['type']
        obj_idx = (eval(obj)['start_idx'], eval(obj)['end_idx'])

        subject_entity.append(sub_entity)
        subject_type.append(sub_type)
        subject_idx.append(sub_idx)

        object_entity.append(obj_entity)
        object_type.append(obj_type)
        object_idx.append(obj_idx)

    out_dataset = pd.DataFrame({'id':dataset['id'], 
                                'sentence':dataset['sentence'],
                                'subject_entity':subject_entity, 'subject_type':subject_type, 'subject_idx':subject_idx,
                                'object_entity':object_entity, 'object_type':object_type, 'object_idx':object_idx,
                                'label':dataset['label'],})
  
    return out_dataset 