import numpy as np

def read_glove():
    with open("/data/glove.6B.50d.txt", 'r') as f:
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

def read_label(X):
    labels=set()
    for i in X:
        labels.add(i)
    label_to_index=dict()
    index_to_lable=dict()
    i=0
    for label in sorted(labels):
        label_to_index[label]=i
        index_to_lable[i]=label
        i+=1
    return label_to_index, index_to_lable

X=['a', 'b','c', 'a']
label_to_index, index_to_lable=read_label(X)
print(label_to_index)
print(index_to_lable)

def one_hot_vector(X):
    lable_to_index,_=read_label(X)
    X_encode=np.asarray([lable_to_index[i] for i in X])
    return np.eye(len(lable_to_index))[X_encode.reshape(-1)]

X=['a', 'b','c', 'a']
print(one_hot_vector(X))

