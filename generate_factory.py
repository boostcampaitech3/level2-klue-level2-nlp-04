import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.cuda.amp import autocast

tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
                                         bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
model = AutoModelForCausalLM.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',pad_token_id=tokenizer.eos_token_id,
    torch_dtype='auto', low_cpu_mem_usage=False).to(device='cuda', non_blocking=True)

_ = model.eval()

def generate(df:pd.DataFrame, SAVE_DIR:str)->None:
    print("화이팅!!!!!!!")
    for i in tqdm(range(len(df))):
        prompt = df['sentence'][i]
        with torch.no_grad():
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
            with torch.cuda.amp.autocast():
                gen_tokens = model.generate(tokens ,do_sample=True, temperature=0.8, max_length = 64)
            generated = tokenizer.batch_decode(gen_tokens)[0]
        df['sentence'][i] = generated
    df.to_csv(SAVE_DIR, encoding = 'utf-8')
    print("종료되었습니다.")

if __name__ == '__main__':

    DATA_DIR = ''
    SAVE_DIR = ''
    df = pd.read_csv(DATA_DIR, index_col=0, encoding='utf-8')
    generate(df, SAVE_DIR)