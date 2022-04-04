import pandas as pd
import re
from typing import List, Dict, Tuple


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


def preprocessing_dataset_v2(dataset: pd.DataFrame) -> pd.DataFrame:
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    이후 Typed Entity Marker with Punctuation와 전처리릊 진행한 이후 DataFrame을 반환힙니다. 
    
    subject_entity (변겅 전) -> subject_entity, subject_type, subject_idx (변경 후)
    {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} (변경 전) -> "비틀즈", "(24, 26)", "ORG" (변경 후)

    전처리
        1. 특수 기호 일반화
        2. 불필요한 특수 기호 제거
        3. 공백 일반화
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

    dataset = typed_entity_marker_with_punctuation(dataset)

    dataset['sentence'] = dataset['sentence'].pipe(clean_punc)\
                                            .pipe(remove_special_char)\
                                            .pipe(remove_repeated_spacing)\
                                            .pipe(remove_repeated_spacing)
    
    dataset['subject_entity'] = dataset['subject_entity'].pipe(clean_punc)\
                                                        .pipe(remove_special_char)\
                                                        .pipe(remove_repeated_spacing)\
                                                        .pipe(remove_repeated_spacing)
    
    dataset['object_entity'] = dataset['object_entity'].pipe(clean_punc)\
                                                        .pipe(remove_special_char)\
                                                        .pipe(remove_repeated_spacing)\
                                                        .pipe(remove_repeated_spacing)
                    
  
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


def tokenized_dataset(df: pd.DataFrame, tokenizer) -> Dict:
    """
    Add Query로 데이터에 추가하여 tokenizer에 따라 sentence를 tokenizing 합니다.
    이 때 subject_entity와 object_entity 순서를 고려하여 Query(question)을 생성합니다. 

    추가 : @*PER*이순신@과 #^POH^무신#의 관계  
    """
    new_question = []
    for idx in range(len(df)):
        row = df.iloc[idx]
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
            question = (
                '@*'
                + subject_type 
                + '*' 
                + subject_entity 
                + '@와 #^' 
                + object_type 
                + '^' 
                + object_entity
                + '#의 관계'
            )
        else:
            question = (
                '#^'
                + subject_type 
                + '^' 
                + subject_entity 
                + '#와 @*' 
                + object_type 
                + '*' 
                + object_entity
                + '@의 관계'
            )
        new_question.append(question)

    tokens = ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']
    tokenizer.add_tokens(tokens)
        
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
    