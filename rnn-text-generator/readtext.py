import numpy as np

def createdict(file="data.txt"):
    dicti={}
    file=open(file)
    say=0
    for i in file.read():
        if i not in dicti:
            dicti[i]=say
            dicti[say]=i
            say=say+1
    np.save("dict.npy",dicti)

def loaddict(file="dict.npy"):
    return np.load(file).item()

def text2data(text=open("data.txt").read(),newdict=True):
    if newdict:
        createdict()
    dicti=loaddict()
    nclasses=int(len(dicti)/2)
    return [dicti[c] for c in text]

def data2text(data):
    dicti=loaddict()
    text=""
    for i in data:
        text=text+dicti[i]
    return text

##np.eye(nclasses)[dicti[c]]
