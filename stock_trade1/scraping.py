from bs4 import BeautifulSoup
import pandas as pd
import requests
from datetime import datetime

year = 2018
f = open('nikkei.txt')

#東証一部上場している企業の銘柄コード
stock_numbers = f.readlines()
f.close()

for number in stock_numbers:
    number = number.rstrip('\n')
    #保存先のディレクトリ
    output = open('./data/{}_{}.txt'.format(number,year), mode='w')

    #スクレイピング先のurl
    url = 'https://kabuoji3.com/stock/{}/{}/'.format(number,year)
    soup = BeautifulSoup(requests.get(url).content,'html.parser')

    tag_tr = soup.find_all('tr')
    #テーブルの各データの取得
    data = []
    for i in range(1,len(tag_tr)):
        data.append([d.text for d in tag_tr[i].find_all('td')])
        #print(data[i-1])
        cnt = 0;
        for j in data[i-1]:
            if( cnt == 0):
                cnt += 1
                continue
            output.write(j)
            output.write('\n')
            cnt += 1
