import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from utilities.main_utilities import *
from dataloader.main_dataloader import *
from dataset.main_dataset import *
from constants import *

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
    Tokenizer_NAME = args.model 
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    path_list = []
    for K in range(args.fold):
        ## load my model
        MODEL_NAME = os.path.join(BEST_MODEL_DIR, f'{args.model_name}{K}') # model dir.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.parameters
        model.to(device)

        ## load test datset
        test_id, test_dataset, test_label = load_test_dataset(TEST_DIR, tokenizer)
        Re_test_dataset = RE_Dataset(test_dataset ,test_label)

        ## predict answer
        pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
        pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
        
        ## make csv file with predicted answer
        #########################################################
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

        os.makedirs(f'./prediction/{args.model_name}', exist_ok=True)
        path = os.path.join(f'./prediction/{args.model_name}', f"submission{K}.csv")
        path_list.append(path)
        output.to_csv(path, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
        #### 필수!! ##############################################
        print('---- Finish! ----')
    
    final_output = voting(path_list)
    path = os.path.join(f'./prediction/{args.model_name}', "submission_final.csv")
    final_output.to_csv(path, index=False)
    print('--------End---------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--model', type=str, default='klue/roberta-large')
    parser.add_argument('--model_name', type=str, default="good")
    args = parser.parse_args()
    print(args)
    main(args)
  