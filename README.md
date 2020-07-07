# Project-Vald

from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import argparse
from pathlib import Path
import datetime
import os
import random
import string
import pandas as pd
from pathlib import Path
#from transformers import BertModel, BertConfig
import re
data=pd.read_excel('/content/to_be_labelled(1).xlsx',nrows=20)[['Tweets','Manual_tags']]
b=[]
for i in data.Tweets:
  a=str(i)
  a=a.replace('!,?,.,&',"")
  a=re.sub(r'@[\w]*','',a)
  a=re.sub(r'http\s','',a)
  a=re.sub('[^A-Za-z]+', " ", a)
  a=re.sub(r'\s+',' ',a)
  b.append(a)
b=[item + '.' for item in b]
c=pd.Series(b)
b_s=pd.DataFrame(c)
b_s.columns=['col1']
bb=[i for i in range(len(b_s))]

b_s.to_csv('try.csv',encoding='utf-8')


output_dir=Path('output/')

model_path=Path('/content/finBERT-master/models/classifier_model/finbert-sentiment/')


if not os.path.exists(output_dir):
    os.mkdir(output_dir)



model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)

random_filename = ''.join(random.choice(string.ascii_letters) for i in range(10))
output = random_filename + '.csv'

print("yooooooooo")

out=pd.concat([i  for i in b_s.apply(lambda row: predict(row['col1'],model),axis=1)])
print(out)
for i in out:
  print(i)


print("this is the original")
print(b_s)
