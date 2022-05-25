from tqdm import tqdm
import torch
from transformers import BertTokenizer
from model import BERTforNER_CRF
# from main import tab,idx2label#写对应的标签集


def test():
# tab = ['PER.NAM', 'PER.NOM', 'LOC.NAM', 'LOC.NOM', 'GPE.NAM', 'GPE.NOM', 'ORG.NAM', 'ORG.NOM']
    # tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
    tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate']
    # tab = ['PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    # tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate', 'PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    labels = ['O']
    for l in tab:
        for seg in ['B', 'I']:
            token = seg + '-' + l
            labels.append(token)

    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    device = "cuda:1" if torch.cuda.is_available() else 'cpu'
    model = BERTforNER_CRF.from_pretrained('./saved_model')
    model.to(device)
    model.eval()

    while True:
        text = input("请输入：")
        # chars = ''.join(text)
        input_ids = tokenizer(text)['input_ids']
        token_type_ids = tokenizer(text)['token_type_ids']
        attention_mask = tokenizer(text)['attention_mask']
        input_ids = torch.unsqueeze(torch.tensor(input_ids, dtype=torch.int64),0)
        token_type_ids =torch.unsqueeze(torch.tensor(token_type_ids, dtype=torch.int64),0)
        attention_mask = torch.unsqueeze(torch.tensor(attention_mask, dtype=torch.int64),0)

        res = model(input_ids, attention_mask)
        label_len = len(res[1])
        res = res[1][1:label_len-1]  ##由于返回的标签是带开始和结束符的，因此要去掉开头和结尾
        pred_labels = [idx2label[ix] for ix in res]

        print([f'{w}_{s}' for w, s in zip(text, pred_labels)])

if __name__ == '__main__':
    test()