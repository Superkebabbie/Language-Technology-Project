#prints a list of all unique tags in the data

def readData(path):
    sents = open(path,'r').read().split('\n\n')
    tagval = []

    for s in sents[:len(sents)-1]:
        lines = s.split('\n')
        sout = []
	
        words = []
        for l in lines:
            try:
                (word,tag) = l.split('\t')
		if (tag not in tagval):
                    tagval.append(tag)
            except:
                pass
    return tagval

readData('data/train.txt')

print(readData('data/train.txt'))
