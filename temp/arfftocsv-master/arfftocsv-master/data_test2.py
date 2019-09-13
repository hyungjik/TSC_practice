import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from sklearn.model_selection import train_test_split

# Samsung
samsung = fdr.DataReader('068270', '2005-07-19', '2019-09-01')
s_c = samsung.reset_index()['Close'].tolist()

data_length = 70  # day length for one sliced data
after1_full = []
after3_full = []
after7_full = []
# 전수저장
s_c_full = []
for i in range(3494-data_length-6):
    s_c_full.append(s_c[0+i: data_length+i])

# 7간격으로 데이터 추출
s_c_7 = []
for i in range(int((3494-data_length-6)/7)):
    s_c_7.append(s_c[0+i*7: data_length+i*7])


# print('s_c_7', len(s_c_7))
normalized = []
for data in s_c_7:
    normalized.append([a/data[-1] for a in data])

normalized = pd.DataFrame(normalized)
# print(normalized)


# 여기 아래는 수정필요
for i in range(int(len(s_c_full)/7)):
    k = i*7
    after1_full.append((s_c[data_length+k]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)
    after3_full.append((s_c[data_length+k+2]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)
    after7_full.append((s_c[data_length+k+6]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)

# after7_full
# print(after7_full)
# print(len(after7_full))

count_3 = pd.DataFrame(columns=[0,1,2], index=[0,1,2])
count_3.loc[:,:] = 0
# count.loc[1,0] += 1
total_full = []
total_full.append(after1_full)
total_full.append(after3_full)
total_full.append(after7_full)
answer = []

i = 0
for per in total_full[i]:
    if abs(per) < 3:
        count_3.loc[i,2] += 1
        answer.append(2)
    elif per > 3:
        count_3.loc[i,0] += 1
        answer.append(0)
    elif per < -3:
        count_3.loc[i,1] += 1
        answer.append(1)
print(count_3)
answer = pd.DataFrame(answer)
# answer

# after7_full

result = pd.concat([answer, normalized], axis=1)
# print(result)

# result.values.tolist()

df_train, df_test = train_test_split(result, test_size=0.5, random_state=0)
print(df_train.shape, df_test.shape)
# print(df_train)
y_train = df_train.values[:, 0]
x_train = df_train.values[:, 1:]
y_test = df_test.values[:, 0]
x_test = df_test.values[:, 1:]
# print(df_train.values[:,0])
# print(df_train.values[:,1:])
