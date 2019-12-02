import pandas as pd
import numpy as np

data=pd.read_csv("data/text_emotion.csv")

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

def preprocessing(text):
    text=str(text)
    text=text.strip()

    #Lower case
    text=text.lower()

    # Remove url:
    text=re.sub(r"http\S+","", text)

    #Remove tag @ and special characters:
    txt=text.split()
    for i, word in enumerate(txt):
        if word[0]=='@':
            txt[i]=''
            continue
        txt[i]=re.sub(r"(can't|cannot)", 'can not', txt[i])
        txt[i]=re.sub(r"n't", ' not', txt[i])
        txt[i]=re.sub(r"'s", ' is', txt[i])
        txt[i]=re.sub(r"'m", ' am', txt[i])
        txt[i]=re.sub(r"'ll", ' will', txt[i])
        txt[i]=re.sub(r'\W+',' ', txt[i])
        txt[i]=txt[i].strip() 


    #Remove repetition :
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    for i, word in enumerate(txt):
        txt[i]=re.sub(pattern, r"\1",txt[i])

    #Removal of lemmas:
    for i, word in enumerate(txt):
        txt[i]=lem.lemmatize(txt[i],'v')
    
    text=' '.join(i for i in txt if i!='')

    return text

def save(data):

    #Maping label in to numeric number
    label={}
    k=0
    for i in data['sentiment']:
        if i not in label:
            label[i]=k
            k+=1

    data['sentiment']=data['sentiment'].map(label)

    #Preprocess text

    data['content']=data['content'].map(lambda x: preprocessing(x))

    #Export to csv file

    data.to_csv(r'data/data_after.csv')

def read_data():
    data=pd.read_csv(r'data/data_after.csv')

    y=np.array(data['sentiment'], dtype=int)
    X=np.array(data['content'])

    return X, y

def convert_to_one_hot(y,c):
    return np.eye(c)[y.reshape(-1)]


def read_glove():
    with open("data/glove.6B.50d.txt", 'r') as f:
        words=set()
        word_to_vec_map=dict()
        for line in f:
            l=line.strip().split()
            words.add(l[0])
            word_to_vec_map=np.array(l[1:], dtype=np.float)
        i=1
        word_to_index=dict()
        index_to_word=dict()
        for word in sorted(words):
            word_to_index[word]=i
            index_to_word[i]=word
            i+=1
        return word_to_vec_map, word_to_index, index_to_word


def convert_string_to_indices(X, word_to_index, max_len):
    X_indices= np.zeros((X.shape[0], max_len))
    for i in range(X.shape[0]):
        j=0
        for word in X[i].split():
            if word in word_to_index:
                X_indices[i,j]=word_to_index[word]
                j+=1
    return X_indices