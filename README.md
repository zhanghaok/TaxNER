# Chinese-NER
Chinese Named Entity Recognition (中文命名实体识别).

BILSTM-CRF and BERT-CRF

## Preparation
### Data Format

Each line contains a character and its label, separated by "\t" or space. Each sentence is followed by a blank line.

```
中	B-LOC
国	I-LOC
很	O
大	O

句	O
子	O
结	O
束	O
是	O
空	O
行	O
```

Modify the `labels` in `main.py` according to your dataset:

labels = ['O', 'B-LOC', 'B-ORG', 'B-T', 'I-LOC', 'I-PER', 'B-PER', 'I-ORG', 'I-T']

## Usage
### **Train**
```
# run bert
python main.py --model bert

# run bert+crf
python main.py --model bert --crf
```

```
# run bilstm
# set the learning rate to 1e-2
python main.py --model bilstm \
        --learning_rate 1e-2 \
        --num_train_epochs 20 \
        --train_batch_size 64 \
        --dev_batch_size 32

# run bilstm+crf
python main.py --model bilstm --crf \
        --learning_rate 1e-2 \
        --num_train_epochs 20 \
        --train_batch_size 64 \
        --dev_batch_size 32
```
### **Test**

```
输入一句话给出预测
python test.py
```

