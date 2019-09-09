import requests

url = "https://api.upbit.com/v1/orderbook"

querystring = {"markets":"KRW-BTC"}

response = requests.request("GET", url, params=querystring)

print(response.text)
