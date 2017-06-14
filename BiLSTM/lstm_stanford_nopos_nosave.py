import numpy as np

np.random.seed(113) #set seed before any keras import
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Merge, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.preprocessing import sequence
from collections import defaultdict

import subprocess
import codecs
import sys
import conlleval

w2i = defaultdict(lambda: len(w2i))
PAD = w2i["<pad>"] # index 0 is padding
UNK = w2i["<unk>"] # index 1 is for UNK
t2i = defaultdict(lambda: len(t2i))
TPAD = w2i["<pad>"]

p2i = defaultdict(lambda: len(p2i))
PPAD = p2i["<pad>"]
PUNK = p2i["<unk>"]

def readInputData(path):
    sents = open(path,'r',encoding='utf-8', errors ='ignore').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        slabel = []
        for l in s.split('\n'):
            try:
                (word,tag) = l.split('\t')
                if word != '':
                    sout.append(word)
            except:
                pass
        out.append(sout)
        
    return out
        

def readLabelData(path):
    sents = open(path,'r',encoding='utf-8', errors ='ignore').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        slabel = []
        for l in s.split('\n'):
            if l != '':
                (word,tag) = l.split('\t')
                if word != '':
                    sout.append(tag)
        out.append(sout)
        
    return out

def readPosData(path):
    sents = open(path,'r',encoding='utf-8', errors ='ignore').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        slabel = []
        for l in s.split('\n'):
            (word,tag) = l.split()
            if word != '':
                sout.append(tag)
        out.append(sout)
        
    return out


def indexToWord(num):
    for key, val in t2i.items():
        if val == num:
            return key


#X_train = readInputData('train.txt')
#y_train = readLabelData('train.txt')
#X_dev = readInputData('testfix.txt')
#y_dev = readLabelData('testfix.txt')

#X_train_pos = readPosData('pos_final_train.txt')
#X_dev_pos = readPosData('testpos.txt')

X_train = readInputData('../data/train.txt')
y_train = readLabelData('../data/train.txt')
X_dev = readInputData('../data/dev.txt')
y_dev = readLabelData('../data/dev.txt')
X_test = readInputData('../data/testfix.txt')
y_test = readLabelData('../data/testfix.txt')

X_train_pos = readPosData('../data/trainpos.txt')
X_dev_pos = readPosData('../data/devpos.txt')
X_test_pos = readPosData('../data/testpos.txt')

#Begin Preprocessing ----------------------

# convert words to indices, taking care of UNKs
X_train_num = [[w2i[word] for word in sentence] for sentence in X_train]
w2i = defaultdict(lambda: UNK, w2i) # freeze
X_dev_num = [[w2i[word] for word in sentence] for sentence in X_dev]
X_test_num = [[w2i[word] for word in sentence] for sentence in X_test]


X_train_pos_num = [[p2i[word] for word in sentence] for sentence in X_train_pos]
p2i = defaultdict(lambda: UNK, p2i) # freeze
X_dev_pos_num = [[p2i[word] for word in sentence] for sentence in X_dev_pos]
X_test_pos_num = [[p2i[word] for word in sentence] for sentence in X_test_pos]


# same for labels/tags
y_train_num = [[t2i[tag] for tag in sentence] for sentence in y_train]
t2i = defaultdict(lambda: UNK, t2i) # freeze
y_dev_num = [[t2i[tag] for tag in sentence] for sentence in y_dev]
y_test_num = [[t2i[tag] for tag in sentence] for sentence in y_test]

np.unique([y for sent in y_train for y in sent ])

num_classes = len(np.unique([y for sent in y_train for y in sent]))

num_labels = len(np.unique([y for sent in y_train for y in sent ]))
y_train_1hot = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes=num_labels) for instance_labels in y_train]
y_dev_1hot  = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes=num_labels) for instance_labels in y_dev]
y_test_1hot  = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes=num_labels) for instance_labels in y_test]

# now a single instance is a 2d object (matrix)

max_sentence_length=max([len(s) for s in X_train] 
                        + [len(s) for s in X_dev]
                        + [len(s) for s in X_test])

X_train_pad = sequence.pad_sequences(X_train_num, maxlen=max_sentence_length, value=PAD)
X_dev_pad = sequence.pad_sequences(X_dev_num, maxlen=max_sentence_length, value=PAD)
X_test_pad = sequence.pad_sequences(X_test_num, maxlen=max_sentence_length, value=PAD)

X_train_pos_pad = sequence.pad_sequences(X_train_pos_num, maxlen=max_sentence_length, value=PPAD)
X_dev_pos_pad = sequence.pad_sequences(X_dev_pos_num, maxlen=max_sentence_length, value=PPAD)
X_test_pos_pad = sequence.pad_sequences(X_test_pos_num, maxlen=max_sentence_length, value=PPAD)

y_train_1hot_pad = sequence.pad_sequences(y_train_1hot, maxlen=max_sentence_length, value=TPAD)
y_dev_1hot_pad = sequence.pad_sequences(y_dev_1hot, maxlen=max_sentence_length, value=TPAD)
y_test_1hot_pad = sequence.pad_sequences(y_test_1hot, maxlen=max_sentence_length, value=TPAD)

#End Preprocessing ------------------------


vocabulary_size = len(w2i)
embeds_size=64

# Construct the bidirectional lstm model with word/pos embeddings
result = Sequential()
result.add(Bidirectional(LSTM(32, return_sequences=True, dropout = 0.0)))
result.add(TimeDistributed(Dense(num_labels))) # TimeDistributed 
result.add(Activation('softmax'))
result.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 100
bestFScore = -1
bestmodel = result

#Select best model obtained on development set
for epoch in range(epochs):
    print("At epoch: " + str(epoch))
    result.fit([X_train_pad], y_train_1hot_pad, epochs = 1, batch_size = 5, validation_data=([X_dev_pad], y_dev_1hot_pad))

    prd = result.predict_classes([X_dev_pad])
    prdtest = result.predict_classes([X_test_pad])

    idx = 0
    idx2 = 0
    length = 0
    with open("output.txt", "w") as f:
        for res in prd[:len(prd)]:
            length = max_sentence_length - len(X_dev_num[idx])
            for val in res[length:]:
                output = X_dev[idx][idx2] + " " + y_dev[idx][idx2] + " " + indexToWord(val)
                f.write(output + "\n")
                idx2 = idx2 + 1

            idx += 1
            idx2 = 0
    fscore = conlleval.giveFScore(['lstm_stanford_pos.py', 'output.txt'])
    if (fscore > bestFScore):
        bestFScore = fscore
        print("Found new best model: saving to model.h5")
        bestmodel = result
        
    idx = 0
    idx2 = 0
    length = 0
    with open("outputtest.txt", "w") as f:
        for res in prdtest[:len(prdtest)]:
            length = max_sentence_length - len(X_test_num[idx])
            for val in res[length:]:
                output = X_test[idx][idx2] + " " + y_test[idx][idx2] + " " + indexToWord(val)
                f.write(output + "\n")
                idx2 = idx2 + 1

            idx += 1
            idx2 = 0
    fscore = conlleval.giveFScore(['lstm_stanford_pos.py', 'outputtest.txt'])

#Test on test set for final results
prd = bestmodel.predict_classes([X_test_pad])
idx = 0
idx2 = 0
length = 0

#Save the output of the program in such a way conlleval accepts it
with open("output_testfinal.txt", "w") as f:
    for res in prd[:len(prd)]:
        length = max_sentence_length - len(X_test_num[idx])
        for val in res[length:]:
            output = X_test[idx][idx2] + " " + y_test[idx][idx2] + " " + indexToWord(val)
            f.write(output + "\n")
            idx2 = idx2 + 1

        idx += 1
        idx2 = 0

#Use the previously generated file to compute the final fscore and the matrix of recall/precision
fscore = conlleval.giveFScore(['lstm_stanford_pos.py', 'output_testfinal.txt'])


