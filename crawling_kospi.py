from bs4 import BeautifulSoup
import urllib.request as req
import sys
import io
import FinanceDataReader as fdr

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# dict = {}
# krx_list = []
# Url = "https://finance.naver.com/sise/entryJongmok.nhn?&page="
# for i in range(1,21):
#     url = Url + str(i)
#     res = req.urlopen(url).read()
#     soup = BeautifulSoup(res, "html.parser")
#
#     kospi_list = soup.select("td.ctg")
#     for name in kospi_list:
#         krx = name.a.get('href')[-6:]
#         dict[krx] = name.string
#         krx_list.append(krx)
#         # print('name : ', name.string)
#         # print('KRX : ', name.a.get('href')[-6:])

# print(krx_list)
# print(len(krx_list))
# print(dict)

stock_data = fdr.DataReader('005930', '2014-10-01', '2018-09-30')
stock_data = fdr.DataReader('005930', '2018-10-01', '2019-09-30')
print(type(stock_data))
