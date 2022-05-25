from tqdm import tqdm
import torch
from transformers import BertTokenizer,BertConfig
from model import BERTforNER_CRF
from data_loader import sequence_padding_for_demo
# from main import tab,idx2label#写对应的标签集

TOKEN_PATH = '/home/zhk/pretrained-model/bert-base-chinese'
max_len = 128

def split_str(s):
    return [ch for ch in s]

def extract(chars,tags):
    """
    chars：一句话 "CLS  张    三   是我们  班    主   任   SEP"
    tags：标签列表[O   B-LOC,I-LOC,O,O,O,B-PER,I-PER,i-PER,O]
    返回一段话中的实体
    """
    result = []
    pre = ''
    w = []
    for idx,tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1] #pre LOC
                w.append(chars[idx])#w 张
        else:
            if tag == f'I-{pre}': #I-LOC True
                w.append(chars[idx]) #w 张三
            else:
                result.append([w,pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
    return [[''.join(x[0]),x[1]] for x in result]

def testModel():

# tab = ['PER.NAM', 'PER.NOM', 'LOC.NAM', 'LOC.NOM', 'GPE.NAM', 'GPE.NOM', 'ORG.NAM', 'ORG.NOM']
    # tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
    # tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate']
    # tab = ['PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate', 'PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    labels = ['O']
    for l in tab:
        for seg in ['B', 'I']:
            token = seg + '-' + l
            labels.append(token)
    idx2label = labels
    num_labels = len(labels)

    tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'


    config = BertConfig.from_pretrained(
        TOKEN_PATH,
        num_labels=num_labels,
        hidden_dropout_prob=0.2)
    model = BERTforNER_CRF.from_pretrained(TOKEN_PATH,
                                           config=config,
                                       use_crf=True)
    model.load_state_dict(torch.load("./save_model_0/bert_crf.pt"))

    model.to(device)
    model.eval()

    while True:
        text = input("请输入：")
        X = [split_str(text)]
        print(X)
        # print(type(X))
        # X = ['增', '值', '税', '1', '0', '%', '一', '般', '纳', '税', '人']
        input_ids,attention_mask,pred_mask = sequence_padding_for_demo(X,tokenizer,max_len)
        # print(f"input_ids;{input_ids}")
        # print(f"attention_masks;{attention_mask}")
        # print(f"pred_mask;{pred_mask}")
        '''
               input_ids:  (batch_size, max_seq_length)
               attention_mask:  (batch_size, max_seq_length)
               token_type_ids:  (batch_size, max_seq_length)
               pred_mask: (batch_size, max_seq_length)
               input_labels: (batch_size, )

               return: (batch_size, max_seq_length), loss
        '''

        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pred_mask = pred_mask.to(device)
        res = model(input_ids,attention_mask,pred_mask = pred_mask)
        res = res[0].tolist()
        pred_labels = [idx2label[ix] for ix in res[0]]

        chars = tokenizer.convert_ids_to_tokens(input_ids[0])
        print([f'{w}_{s}' for w, s in zip(text, pred_labels)])
        pred_entities = extract(text,pred_labels)
        print("预测的实体：%s"%pred_entities)
        # for w, s in zip(text, pred_labels):
        #     print(f"{w}_{s}")

def batch_predict():
    print("批量预测")
    pass

if __name__ == '__main__':

    testModel()