import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from utilities.main_utilities import *

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

def num_2_label(n):
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    origin_label = dict_num_to_label[n]
    return origin_label

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
              )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def main(args):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    for K in range(args.fold):
        ## load my model
        MODEL_NAME = os.path.join(args.model_dir,args.wandb_name + f'{K}') # model dir.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.parameters
        model.to(device)

        ## load test datset
        test_dataset_dir = "../dataset/test/test_data.csv"
        test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
        Re_test_dataset = RE_Dataset(test_dataset ,test_label)

        ## predict answer
        pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
        pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
        
        ## make csv file with predicted answer
        #########################################################
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

        output.to_csv(f'./save_prediction/submission{K}.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
        #### 필수!! ##############################################
        print('---- Finish! ----')
    
    df1 = pd.read_csv(f'./save_prediction/submission0.csv')
    df2 = pd.read_csv(f'./save_prediction/submission1.csv')
    df3 = pd.read_csv(f'./save_prediction/submission2.csv')
    df4 = pd.read_csv(f'./save_prediction/submission3.csv')
    df5 = pd.read_csv(f'./save_prediction/submission4.csv')

    df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.2) + df2['probs'].apply(lambda x : to_nparray(x)*0.2) + df3['probs'].apply(lambda x : to_nparray(x)*0.2) + df4['probs'].apply(lambda x : to_nparray(x)*0.2) + df5['probs'].apply(lambda x : to_nparray(x)*0.2)

    for i in range(len(df1['probs'])):
        df1['probs'][i] = F.softmax(torch.tensor(df1['probs'][i]), dim=0).detach().cpu().numpy()
    
    df1['pred_label'] = df1['probs'].apply(lambda x : num_2_label(np.argmax(x)))
    df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))

    df1.to_csv(f'./save_prediction/submission_final.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default="./best_model")
    parser.add_argument('--model_name', type=str, default="klue/bert-base")
    parser.add_argument('--wandb_name', type=str, default= 'kfold(not stratify)')
    args = parser.parse_args()
    print(args)
    main(args)
  
