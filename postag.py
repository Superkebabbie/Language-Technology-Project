#generate a sister file for some input data file where every token has a POS-tag

input = 'data/testfix.txt'
output= 'data/testpos.txt'

import nltk, os
from nltk.tag import StanfordPOSTagger
tagger = StanfordPOSTagger('Stanford Tagger/english-bidirectional-distsim.tagger',path_to_jar='Stanford Tagger/stanford-postagger.jar')

os.environ['JAVAHOME'] = "C:/Program Files/Java/jre1.8.0_131/bin/java.exe"
nltk.internals.config_java("C:/Program Files/Java/jre1.8.0_131/bin/java.exe")

def readData(path):
    sents = open(path,'r',encoding='utf-8', errors ='ignore').read().split('\n\n')
    out = []
    for s in sents[:]:
        sout = []
        for l in s.split('\n'):
            try:
                (word,tag) = l.split('\t')
                if word != '':
                    sout.append(word)
            except:
                pass
        out.append(sout)
    return out
    
def postag(sents):
    postags = []
    idx = 0
    for s in sents:
        print("%d/%d"%(idx,len(sents)))
        idx += 1
        try:
            postags.append([(word,tag) for word,tag in tagger.tag(s)])
        except OSError:
            print("OSError in sequence \"%s\""%(' '.join(s)))
            postags.append([(w, 'UNK') for w in s])
    return postags
    
def writePos(postags,output):
    file = open(output,'w',encoding='utf-8')
    for s in postags:
        for w in s:
            file.write("%s %s\n"%(w[0],w[1]))
        file.write('\n')
    file.close()
    
sents = readData(input)
print("Read data!")
postags = postag(sents)
print("Tagged data!")
writePos(postags,output)
print("Done!")