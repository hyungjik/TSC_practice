import pyupbit
# print(pyupbit.Upbit)

# 상장된 시장 조회
# tickers = pyupbit.get_tickers()
# print(tickers)

# tickers = pyupbit.get_tickers(fiat="KRW")
# print(tickers)

# price = pyupbit.get_current_price("KRW-XRP")
# print(price)

# df = pyupbit.get_ohlcv("KRW-BTC")
# print(df)

df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1")
# print(df)

print(pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=500))
