import pickle
import pandas as pd

def save_preprocessed_data(save_path:str, data:pd.DataFrame):
    """ 전처리된 데이터프레임을 피클 파일로 저장합니다. """
    file = open(save_path, "wb")
    pickle.dump(data, file)
    file.close()

def load_preprocessed_data(save_path:str):
    """ 전처리된 데이터프레임을 피클 파일을 통해 불러옵니다. """
    f = open(save_path, "rb")
    data = pickle.load(f)
    f.close()
    return data