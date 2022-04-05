import re
import pandas as pd
from tqdm import tqdm


def load_generate_data()-> pd.DataFrame:

    dataset_dir = '/opt/ml/code/augmentation/gen_train_ing.csv'

    pd_dataset = pd.read_csv(dataset_dir, index_col = 0, encoding='utf-8')
    dataset = preprocessing_generate_dataset(pd_dataset.iloc[:2999])

    return dataset

def preprocessing_generate_dataset(df:pd.DataFrame)->pd.DataFrame:
    
    df = df.pipe(end_word_filtering)\
           .pipe(entity_spacing_filtering)\
           .pipe(bad_char_filtering)\
           .pipe(short_sentence_filtering)\
           .pipe(temp_filtering)

    subject_entity = []
    object_entity = []
    source = []
    for sub_e, sub_i, sub_t, obj_e, obj_i, obj_t in zip(df['subject_entity'], df['subject_idx'], df['subject_type'], df['object_entity'], df['object_idx'], df['object_type']):
        
        sub_start = int(sub_i[1:-1].split(',')[0])
        sub_end = int(sub_i[1:-1].split(',')[1])
        
        obj_start = int(obj_i[1:-1].split(',')[0])
        obj_end = int(obj_i[1:-1].split(',')[1])

        subject_entity.append(str(dict(word=sub_e, start_idx=sub_start, end_idx=sub_end, type=sub_t)))
        object_entity.append(str(dict(word=obj_e, start_idx=obj_start, end_idx=obj_end, type=obj_t)))
        source.append('generate')

    df = pd.DataFrame({'id':df['id'],
                        'sentence':df['sentence'],
                        'subject_entity':subject_entity,
                        'object_entity':object_entity,
                        'label':df['label'],
                        'source':source})

    return df

def end_word_filtering(df:pd.DataFrame)->pd.DataFrame:

    ending_list = ["다.","요.","이다","입니다",'선다', '가다', '폈다', '왔다', '셨다', '하다', '째다', '찮다', '웠다', '위다',  
               '온다', '겁다', '제다', '한다', '준다', '도다', '댔다', '’다', '뉜다', '뤘다', '른다', '만다', 
               '눈다', '건다', '렀다', '난다', '룬다', '겼다', '친다', '않다', '쳤다', '는다', '푼다', '헀다', 
               '낀다', '췄다', '사다', '썼다', '부다', '본다', '쓴다', '맞다', '지다', '둔다', '린다', '샀다', 
               '컸다', '기다', '넸다', '적다', '서다', '란다', '태다', '졌다', '연다', '많다', '화다', '깝다', 
               '이다', '안다', '퍼다', '았다', '렸다', '끈다', '눴다', '간다', '켰다', '몄다', '크다', '였다', 
               '꿨다', '례다', '긴다', '르다', '뒀다', '갔다', '힌다', '신다', '랐다', '버다', '킨다', '체다', 
               '없다', '됐다', '있다', '터다', '짙다', '운다', '렇다', '냈다', '우다', '치다', '주다', '든다', 
               '꾼다', '인다', '섰다', '스다', '봤다', '깊다', '녔다', '트다', '같다', '리다', '뉴다', '노다', 
               '자다', '싸다', '빴다', '툰다', '혔다', '었다', '진다', '니다', '놨다', '좋다', '챈다', '떴다', 
               '유다', '나다', '민다', '났다', ')다', '탔다', '탰다', '했다', '텁다', '낸다', '된다', '줬다',
               '쁘다', '선다', '가다', '저다', '폈다', '왔다', '셨다', '해다', '하다', '째다', '겠다', '빠다', 
               '찮다', '웠다', '온다', '겁다', '제다', '한다', '준다', '도다', '높다', '댔다', '’다', '뉜다', 
               '뤘다', '만다', '른다', '고다', '녀다', '재다', '회다', '건다', '대다', '렀다', '난다', '룬다', 
               '드다', '계다', '겼다', '친다', "'다", '않다', '투다', '랜다', '쳤다', '때다', '즈다', '는다', 
               '%다', '췄다', '사다', '썼다', '찼다', '부다', '본다', '쓴다', '땄다', '거다', '뗀다', '카다', 
               '지다', '둔다', '린다', '넸다', '컸다', '샀다', '기다', '챘다', '서다', '죄다', '란다', '과다', 
               '꼈다', '태다', '졌다', ' 다', '퉜다', '랬다', '연다', '화다', '다다', '많다', '깝다', '오다', 
               '이다', '퍼다', '뗐다', '닌다', '았다', '댄다', '뜬다', '렸다', '눴다', '끈다', '간다', '내다', 
               '더다', '미다', '쇼다', '켰다', '몄다', '교다', '크다', '였다', '쳣다', '래다', '꿨다', '볐다', 
               '여다', '례다', '긴다', '뎠다', '르다', '뒀다', '갔다', '렵다', '힌다', '랐다', '버다', '킨다', 
               '체다', 's다', '없다', '됐다', '있다', '터다', '운다', '으다', '냈다', '우다', '은다', '주다', 
               '탠다', '처다', '든다', '레다', '꾼다', '인다', '춘다', '엿다', '섰다', '뀐다', '커다', '스다', 
               '핀다', '구다', '아다', '깊다', '봤다', '녔다', '트다', '같다', '리다', '자다', '씨다', '세다', 
               '호다', '그다', '툰다', '파다', '혔다', '띈다', '빈다', '었다', '진다', '니다', '잇다', '놨다', 
               '초다', '러다', '타다', '나다', '딴다', '민다', '났다', '깼다', ')다', '탰다', '탔다', '차다', 
               '했다', '셌다', '낸다', '된다', '표다', '수다', '줬다']
    
    bad_sentence = []
    for i in tqdm(range(len(df))) :
        flag = 0
        for end_word in ending_list :
            if end_word in df['sentence'][i] :
                flag = 1
        
        # bad sentence
        if flag == 0 :
            bad_sentence.append(i)
        
        # good sentence
        if flag == 1 :
            # 2. dot slicing
            dot_cnt = df['sentence'][i].count(".")
            if dot_cnt >= 1 :
                first_dot_idx = df['sentence'][i].find(".")
                
                if dot_cnt >= 2 :
                    second_dot_idx = df['sentence'][i].find(".", first_dot_idx + 1)
                    
                    try : 
                        df['sentence'][i] = df['sentence'][i][:second_dot_idx+1]
                    except :
                        df['sentence'][i] = df['sentence'][i][:first_dot_idx+1]
                    
                else :
                    df['sentence'][i] = df['sentence'][i][:first_dot_idx+1]
    
    df = df.drop(bad_sentence)
    df['id'] = [i for i in range(len(df))]
    df.reset_index(drop=False, inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    
    return df

def entity_spacing_filtering(df:pd.DataFrame)->pd.DataFrame:

    bad_sentence = []

    for i in tqdm(range(len(df))):
        sub_i = df['subject_idx'][i][1:-1].split(',')
        obj_i = df['object_idx'][i][1:-1].split(',')
        
        if min(int(sub_i[1]), int(obj_i[1])) + 60 < max(int(sub_i[0]), int(obj_i[0])) :
            bad_sentence.append(i)

    df = df.drop(bad_sentence)
    df['id'] = [i for i in range(len(df))]
    df.reset_index(drop=False, inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    return df

def bad_char_filtering(df:pd.DataFrame)->pd.DataFrame:

    bad_sentence = []
    bad_list = ['?', '@', '—', '©', '\n\n']

    for i in tqdm(range(len(df))) :
        flag = 0
        for bad_word in bad_list:
            if bad_word in df['sentence'][i]:
                flag = 1
                bad_sentence.append(i)
                break

    df = df.drop(bad_sentence)
    df['id'] = [i for i in range(len(df))]
    df.reset_index(drop=False, inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    
    return df

def short_sentence_filtering(df:pd.DataFrame)->pd.DataFrame:
    
    bad_sentence = []

    for i in tqdm(range(len(df))):
        if len(df['sentence'][i]) < 20:
            bad_sentence.append(i)
     
    df = df.drop(bad_sentence)
    df['id'] = [i for i in range(len(df))]
    df.reset_index(drop=False, inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    
    return df

def temp_filtering(df:pd.DataFrame)->pd.DataFrame:
    
    bad_sentence = []

    for i in tqdm(range(len(df))) :
        df['sentence'][i] = re.sub('(([-=])\\2{2,})', '', df['sentence'][i])
        
        line_cnt = df['sentence'][i].count("\n")
        
        if line_cnt >= 3 :
            bad_sentence.append(i)
        
        elif "\n" in df['sentence'][i] :
            df['sentence'][i] = df['sentence'][i].replace("\n", " ")
            # print("\n")

    df = df.drop(bad_sentence)
    df['id'] = [i for i in range(len(df))]
    df.reset_index(drop=False, inplace=True)
    df.drop(['index'], axis=1, inplace=True)
   
    return df