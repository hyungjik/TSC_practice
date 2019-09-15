# below codes are from https://github.com/hfawaz/dl-4-tsc (19/09/15)
# I added some steps and modified for finance daily ohlcv data
# for Time Series Classification

from utils.util_my import read_datasets
from utils.util_my import univariate_data_preprocessing
from utils.util_my import create_directory
from utils.util_my import transform_labels
from utils.util_my import create_classifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import sklearn
import os

# ----------------------------------------------------------------------------
# SELECT stock : samsung, lg, naver, kakao, kospi200
stock_name = 'samsung'
input_stock = read_datasets(stock_name)
# SELECT Length and Distance
data_length = 70  # day length for one sliced data
gap_distance = 7  # gap of time series data
# SELECT column : 'Open', 'Close', 'High', 'Low', 'Volume'
column_name = 'Close'
# SELECT what percentage you define increase and decrease boundary
percent_boundary = 3
# SELECT algorithm : 'fcn','mlp','resnet','cnn','mcnn','mcdcnn','twiesn','tlenet','encoder'
classifier_name = 'mlp'

# create result folder
analysis_datasets = 'univariate'
root_dir = os.getcwd()
dataset_name = stock_name+'_'+column_name+'_'+str(data_length)+'_'+str(gap_distance)+'_'+str(percent_boundary)
print('dataset_name', dataset_name)
output_directory = root_dir+'/'+'results'+'/'+analysis_datasets+'/'+classifier_name+'/'+dataset_name+'/'
output_directory = create_directory(output_directory)

# preprocessing finance data for deep learning algorithm
final_data = univariate_data_preprocessing(input_stock, column_name, data_length, gap_distance, percent_boundary)

df_train, df_test = train_test_split(final_data, test_size=0.5, random_state=0)
y_train = df_train.values[:, 0]
x_train = df_train.values[:, 1:]
y_test = df_test.values[:, 0]
x_test = df_test.values[:, 1:]


datasets_dict = {}

datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                               y_test.copy())

x_train = datasets_dict[dataset_name][0]
y_train = datasets_dict[dataset_name][1]
x_test = datasets_dict[dataset_name][2]
y_test = datasets_dict[dataset_name][3]

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
y_train, y_test = transform_labels(y_train, y_test)

# save orignal y because later we will use binary
y_true = y_test.astype(np.int64)
# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder()
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

if len(x_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_train.shape[1:]

classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
classifier.fit(x_train, y_train, x_test, y_test, y_true)
