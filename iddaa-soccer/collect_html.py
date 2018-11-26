import urllib.request
import bs4 as bs


def mac(url):
    try:
        html=urllib.request.urlopen(url)
        s = bs.BeautifulSoup(html,"html.parser")
        if "İspanya Primera" in s.find(class_="match-info-wrapper-season").text:
            shtml=str(s)
            with open('matchs/'+url[46:53]+'.html','w',encoding='utf-8') as f:
                f.write(shtml)
            print("Saved: "+url)

def main(x,y):
    for i in range(x,y):
        mac("http://www.mackolik.com/Match/Default.aspx?id="+str(i))


main(2874500,2875500)
main(2575000,2576000)
main(2134250,2135250)
main(1361000,1362000)
