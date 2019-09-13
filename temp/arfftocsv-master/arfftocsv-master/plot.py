import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# Samsung
samsung = fdr.DataReader('068270', '2005-07-19', '2019-09-01')

# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
#
# samsung['Close'][0:70].plot()
# tmp = samsung['Close'][0:70]
# plot_test = [a/tmp[-1] for a in tmp]
# # plt.plot(plot_test)
# plt.show()


s_c = samsung.reset_index()['Close'].tolist()


data_length = 70  # day length for one sliced data
after1_full = []
after3_full = []
after7_full = []
# 전수저장
s_c_full = []
p = 0
for p in range(3494-data_length-6):
    s_c_full.append(s_c[0+p: data_length+p])
    p += 1

# len(s_c_full)

# 여기 아래는 수정필요
for i in range(int(len(s_c_full)/7)):
    k = i*7
    after1_full.append((s_c[data_length+k]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)
    after3_full.append((s_c[data_length+k+2]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)
    after7_full.append((s_c[data_length+k+6]-s_c_full[k][-1]) / s_c_full[k][-1] * 100)

# after7_full
# print(after7_full)
print(len(after7_full))

count_3 = pd.DataFrame(columns=[0,1,2], index=[0,1,2])
count_3.loc[:,:] = 0
# count.loc[1,0] += 1
total_full = []
total_full.append(after1_full)
total_full.append(after3_full)
total_full.append(after7_full)

for i in range(len(total_full)):
    for per in total_full[i]:
        if abs(per) < 3:
            count_3.loc[i,2] += 1
        elif per > 3:
            count_3.loc[i,0] += 1
        elif per < -3:
            count_3.loc[i,1] += 1
print(count_3)
num_bins = 50
n, bins, patches = plt.hist(after7_full, num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.show()
