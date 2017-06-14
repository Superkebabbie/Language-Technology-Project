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
    
data = readLabelData('data/testfix.txt')
labs = []
for s in data:
    labs.extend(s)
print(labs)
numO = sum([1 if l == 'O' else 0 for l in labs])
print("%d/%d"%(numO,len(labs)))