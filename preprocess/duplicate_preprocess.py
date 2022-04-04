import pandas as pd

def remove_duplicate_row(df: pd.DataFrame) -> None:
    """
    mislabel된 row를 제거한 clean_train.csv를 생성합니다
    """
    
    clean_df = df.drop([6749, 8364, 11511, 277, 10202, 4212]) # mislabeled row
    clean_df = clean_df.drop_duplicates(['sentence','subject_entity','object_entity', 'label'], keep='first', inplace=False, ignore_index = True)
    clean_df.to_csv("/opt/ml/dataset/train/clean_train.csv", encoding= 'utf-8', index=False)
