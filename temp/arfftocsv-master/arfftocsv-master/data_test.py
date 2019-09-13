import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#
# data = scipy.io.loadmat("Libras.mat")
# data = data['mts']
# flat = np.array(data).flatten()
# print(len(flat))
#
# flat = [item for sublist in flat for item in sublist]
# # print(len(flat)) nbmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
# d0 = [item for sublist in flat[0] for item in sublist]
# # print(d0)
# data = pd.DataFrame(data)
# print(data)


def read(filename):
    data = np.loadtxt(filename)  # delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    # print(data)
    return X, Y


# dataset_name = 'Coffee'
# datasets_dict = {}
#
x_train, y_train = read('Coffee_TRAIN.txt')
# x_test, y_test = read('Coffee_TEST.txt')
# print(len(x_test), len(y_test))
print(x_train.shape, y_train.shape)
print(y_train)
print(x_train)
# datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
#                                y_test.copy())
#
# x_train = datasets_dict[dataset_name][0]
# y_train = datasets_dict[dataset_name][1]
# x_test = datasets_dict[dataset_name][2]
# y_test = datasets_dict[dataset_name][3]


# df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
# df_train.shape, df_test.shape
