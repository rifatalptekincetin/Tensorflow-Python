import numpy as np

def selectionsort(d):
    for i in range(len(d)):
        for j in range(len(d),i,-1):
            currdate=d[j-1][6][0]+d[j-1][6][1]*50+d[j-1][6][2]*1000
            if j == len(d):
                select=d[j-1]
                index=j-1
                date=currdate
            elif currdate<date:
                select=d[j-1]
                index=j-1
                date=currdate
        d[index]=d[i]
        d[i]=select
    return d

data=np.load("data/data.npy")
ingiltere=[x for x in data if x[8]=='in']
almanya=[x for x in data if x[8]=='al']
ispanya=[x for x in data if x[8]=='is']

sortin=selectionsort(ingiltere)
sortal=selectionsort(almanya)
sortis=selectionsort(ispanya)

np.save("data/ingiltere-"+str(len(ingiltere))+".npy",selectionsort(ingiltere))
print("data/ingiltere-"+str(len(ingiltere))+".npy Saved")
np.save("data/ispanya-"+str(len(ispanya))+".npy",selectionsort(ispanya))
print("data/ispanya-"+str(len(ispanya))+".npy Saved")
np.save("data/almanya-"+str(len(almanya))+".npy",selectionsort(almanya))
print("data/almanya-"+str(len(almanya))+".npy Saved")
