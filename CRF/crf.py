import sys, os
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

print(sklearn.__version__)

# errors because of unknown utf characters, and try except since format can be incorrect (so do not add)
def readData(path):
    sents = open(path,'r',encoding='utf-8', errors ='ignore').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        for l in s.split('\n'):
            if l!='':
                (word,tag) = l.split('\t')
                if word != '':
                    sout.append((word,tag))
        out.append(sout)
    return out
    
def readPOS(path):
    sents = open(path,'r',encoding='utf-8').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        for l in s.split('\n'):
            if l!='':
                (word,tag) = l.split()
                sout.append((word,tag))
        out.append(sout)
    return out

train_sents = readData('../data/train.txt')
test_sents = readData('../data/testfix.txt')
train_pos = readPOS('../data/trainpos.txt')
test_pos  = readPOS('../data/testpos.txt')
#train_pos = [[(w,'') for w in s] for s in train_sents] #use to disable pos
#test_pos = [[(w,'') for w in s] for s in test_sents]

print(len(train_sents))
print(len(train_pos))
print(len(test_sents))
print(len(test_pos))

assert(len(train_sents)==len(train_pos))
assert(len(test_sents)==len(test_pos))

def word2features(sequence, postags, i):
    global tagger
    word = sequence[i]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postags[i],
        'postag[:2]=' + postags[i][:2],
        #'word.ishashtag=%s' % (word[0] is '#'),
        #'word.isagent=%s' % (word[0] is '@'),
    ]
    if i > 0:#word has a preceding word
        word1 = sequence[i-1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postags[i-1],
            '-1:postag[:2]=' + postags[i-1][:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sequence)-2:#word has a succeeding word
        word1 = sequence[i+1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postags[i+1],
            '+1:postag[:2]=' + postags[i+1][:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent,postags):
    sequence = [word for word,label in sent]
    if len(sequence)!=len(postags):
        print(sequence)
        print(postags)
        quit()
    postags = [tag for word,tag in postags]
    return [word2features(sequence, postags, i) for i in range(len(sequence))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(train_sents[s],train_pos[s]) for s in range(len(train_sents))]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(test_sents[s],test_pos[s]) for s in range(len(test_sents))]
y_test = [sent2labels(s) for s in test_sents]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.params()

trainer.train('conll2002-esp.crfsuite')


trainer.logparser.last_iteration

print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

conlltagger = pycrfsuite.Tagger()
conlltagger.open('conll2002-esp.crfsuite')

y_test = [sent2labels(s) for s in test_sents]

example_sent = test_sents[0]

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
    
def writeResults(filename,sents,true,pred):
    file = open(filename,'w',encoding='utf-8')
    for sent in range(len(sents)):
        s = sents[sent]
        for word in range(len(s)):
            file.write('%s %s %s\n'%(sents[sent][word][0],true[sent][word],pred[sent][word]))
    file.close()

y_pred = [conlltagger.tag(xseq) for xseq in X_test]

print("commencing serialisation")
writeResults('crf_classified.txt',test_sents,y_test,y_pred)

print(bio_classification_report(y_test, y_pred))

from collections import Counter
info = conlltagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])




