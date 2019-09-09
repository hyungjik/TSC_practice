import FinanceDataReader as fdr

# Samsung
samsung = fdr.DataReader('068270', '2005-07-19', '2019-09-01')
# LG
lg = fdr.DataReader('066570', '2005-07-19', '2019-09-01')
# 네이버
naver = fdr.DataReader('035420', '2005-07-19', '2019-09-01')
# 카카오
kakao = fdr.DataReader('035720', '2005-07-19', '2019-09-01')
# KOSPI200 ETF
kospi200 = fdr.DataReader('KS200', '2005-07-19', '2019-09-01')

print(samsung.head())
print(samsung.tail())
print(len(samsung))
print(lg.head())
print(lg.tail())
print(len(lg))
print(kakao.head())
print(kakao.tail())
print(len(kakao))
print(naver.head())
print(naver.tail())
print(len(naver))
print(kospi200.head())
print(kospi200.tail())
print(len(kospi200))
