import pandas as pd
import inbeom_preprocess

def remove_duplicate_row() -> None:
    """
    mislabel된 row를 제거한 clean_train.csv를 생성합니다
    """
    df = pd.read_csv("/opt/ml/dataset/train/train.csv") # 원본 train.csv
    i_df = inbeom_preprocess.preprocessing_dataset_v2(df)
    i_clean_df =i_df.drop([6749, 8364, 11511, 277, 10202, 4212]) # mislabeled row
    i_clean_df = i_clean_df.drop_duplicates(['sentence','subject_entity','object_entity', 'label'], keep='first', inplace=False, ignore_index = True) # 중복 row 제거
    clean_id = i_clean_df['id']
    clean_df = df[df['id'].isin(clean_id)]
    clean_df.reset_index(drop = True, inplace=True)

    clean_df.to_csv("/opt/ml/dataset/train/clean_train.csv", encoding= 'utf-8', index=False) # 중복 제거한 df
    i_clean_df.to_csv("/opt/ml/dataset/train/i_clean_train.csv", encoding= 'utf-8', index=False) # 중복 제거하고 inbeom_preprocess.preprocessing_dataset_v2로 reshape한 df
