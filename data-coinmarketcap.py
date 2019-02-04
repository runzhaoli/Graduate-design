# encoding:UTF-8
import urllib.request
import pandas as pd


url = "https://api.coinmarketcap.com/v1/ticker/"
data = urllib.request.urlopen(url).read()
data = data.decode('UTF-8')
print(data)

# 把json格式的数据转换成dataframe
dataCopy = pd.read_json(data,typ='frame')
print(dataCopy)
# 存到本地
dataCopy.to_csv('coinmarketcap20171220.csv')