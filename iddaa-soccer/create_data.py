import bs4 as bs
import numpy as np
import os

data=[]
msdata=[]
dsay=0

print(len(os.listdir('matchs')))

for match in os.listdir('matchs'):
    with open('matchs/'+match,'r',encoding='utf-8') as f:
        s = bs.BeautifulSoup(f,"html.parser")
    if "Almanya" in s.find(class_="match-info-wrapper-season").text:
        ulke='al'
    elif "İspanya" in s.find(class_="match-info-wrapper-season").text:
        ulke='is'
    else:
        ulke='in'
    lines=str(s).split('\n')
    for i in lines:
        if "homePlayerList =" in i:
            hp=[x for x in i.replace('[',']').replace('{',']').replace('}',']').split(']') if len(x)>50]
        if "awayPlayerList =" in i:
            ap=[x for x in i.replace('[',']').replace('{',']').replace('}',']').split(']') if len(x)>50]
    hpid=[int(i.replace(":",",").split(",")[1]) for i in hp]
    apid=[int(i.replace(":",",").split(",")[1]) for i in ap]
    htid=int(str(s.find(class_="left-block-team-name")).split('/')[4])
    atid=int(str(s.find(class_="r-left-block-team-name")).split('/')[4])
    rlgl=str(s.find(class_="r-last-games-temp")).split('\n')
    rlg=[]
    for i in rlgl:
        if "M.png" in i:
            rlg.append(-1)
        if "B.png" in i:
            rlg.append(0)
        if "G.png" in i:
            rlg.append(1)
    llgl=str(s.find(class_="last-games-temp")).split('\n')
    llg=[]
    for i in llgl:
        if "M.png" in i:
            llg.append(-1)
        if "B.png" in i:
            llg.append(0)
        if "G.png" in i:
            llg.append(1)
    tarih=s.find(class_="match-info-date").text.replace('.',' ').split(' ')[2:5]
    tarih=[int(tarih[0]),int(tarih[1]),int(tarih[2])]
    ms=s.find(class_="match-score").text.replace(" ","").replace("\n", "").replace("\r","").split("-")
    msdata.append(ms)
    ms=[int(ms[0]),int(ms[1])]
    if(ms[0]==ms[1]):
        ms=[1,0,0]
    elif(ms[0]>ms[1]):
        ms=[0,1,0]
    elif(ms[0]<ms[1]):
        ms=[0,0,1]
    msdata.append(ms)
    if(htid and atid and tarih and len(llg)==5 and len(rlg)==5 and ms):
        if(len(hpid)<11):
            for i in range(len(hpid),11):
                hpid.append(1)
        if(len(apid)<11):
            for i in range(len(apid),11):
                apid.append(1)
        data.append([[htid],hpid[:11],llg,[atid],apid[:11],rlg,tarih,ms,ulke])
        dsay += 1
        print(dsay)
np.save('data/data.npy',data)
print('data/data.npy Saved')
