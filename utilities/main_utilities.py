import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle as pickle
from typing import Callable, Dict, List
import torch
import torch.nn.functional as F

# from criterion import *
# from metric import *
# from optimizer import *
# from scheduler import *
# from tool import *

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred:Callable[[], Dict]):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'f1': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label:List[str])->List[int]:
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    
    return num_label

def num_to_label(label:List[int])->List[str]:
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('/opt/ml/code/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    
    return origin_label

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

def num_2_label(n):
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    origin_label = dict_num_to_label[n]
    
    return origin_label

def voting(path_list):
    df = []

    for path in path_list:
        df.append(pd.read_csv(path))
    df[0]['probs'] = df[0]['probs'].apply(lambda x : to_nparray(x)*0.2) + df[1]['probs'].apply(lambda x : to_nparray(x)*0.2) + df[2]['probs'].apply(lambda x : to_nparray(x)*0.2) + df[3]['probs'].apply(lambda x : to_nparray(x)*0.2) + df[4]['probs'].apply(lambda x : to_nparray(x)*0.2)

    for i in range(len(df[0]['probs'])):
        df[0]['probs'][i] = F.softmax(torch.tensor(df[0]['probs'][i]), dim=0).detach().cpu().numpy()
    
    df[0]['pred_label'] = df[0]['probs'].apply(lambda x : num_2_label(np.argmax(x)))
    df[0]['probs'] = df[0]['probs'].apply(lambda x : str(list(x)))
    
    return df[0]
