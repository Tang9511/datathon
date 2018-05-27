from nltk.tokenize import sent_tokenize,word_tokenize
import pandas as pd
import re
'''
file="pro_req.csv"
with open(file, 'rb') as f:
  contents = f.read()

'''
file="prog_req.csv"
contents=pd.read_csv(file)
#contents='Programme Code: JS3466",34,Any Best 5 Subjects,"English Language,Mathematics",,"2015-16: 22.7 (4X + 1 Best Elective),2016-17: 23.2 (Any Best 5 Subjects),2017-18: 22.6 (Any Best 5 Subjects)"'
contents = contents.replace(","," ").replace(":"," ")
word=word_tokenize(contents.lower())
print(word)

#dse=csv['dse']
#print(dse.iloc[1])