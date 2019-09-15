from utils.util_my import read_close_datasets

import scipy.io
import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.preprocessing import LabelEncoder
import FinanceDataReader as fdr
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# SELECT stock : samsung, lg, naver, kakao, kospi200
stock_name = 'samsung'
input_stock = read_close_datasets(stock_name)
# SELECT column : 'Open', 'Close', 'High', 'Low', 'Volume'
column_name = 'Close'
# SELECT Length and Distance
data_length = 70  # day length for one sliced data
gap_distance = 7  # gap of time series data


# univariate learning with 'Close' data
close_raw = input_stock.reset_index()['Close'].tolist()

after1_full = []
after3_full = []
after7_full = []

# univariate time series data for learnings
close_data = []
for i in range(len(input_stock)-data_length-6):
    close_data.append(close_raw[0+i: data_length+i])

# univariate time series data with gap_distance for learnings
close_divided = []
for i in range(int((len(input_stock)-data_length-6)/gap_distance)):
    close_divided.append(close_raw[0+i*gap_distance : data_length+i*gap_distance])


# print('close_divided', len(close_divided))
normalized = []
for data in close_divided:
    normalized.append([a/data[-1] for a in data])

normalized = pd.DataFrame(normalized)
# print(normalized)


# 여기 아래는 수정필요
for i in range(int(len(close_data)/gap_distance)):
    k = i*gap_distance
    after1_full.append((close_raw[data_length+k]-close_data[k][-1]) / close_data[k][-1] * 100)
    after3_full.append((close_raw[data_length+k+2]-close_data[k][-1]) / close_data[k][-1] * 100)
    after7_full.append((close_raw[data_length+k+6]-close_data[k][-1]) / close_data[k][-1] * 100)

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
# print(count_3)
answer = pd.DataFrame(answer)
# answer

# after7_full

result = pd.concat([answer, normalized], axis=1)
# print(result)

# result.values.tolist()

df_train, df_test = train_test_split(result, test_size=0.5, random_state=0)
# print(df_train.shape, df_test.shape)
# print(df_train)
y_train = df_train.values[:, 0]
x_train = df_train.values[:, 1:]
y_test = df_test.values[:, 0]
x_test = df_test.values[:, 1:]
# print(df_train.values[:,0])
# print(df_train.values[:,1:])

#------------------------------------------------------------------------------

# data = scipy.io.loadmat("Libras.mat")
# data = data['mts']
# flat = np.array(data).flatten()
# print(len(flat))
#
# flat = [item for sublist in flat for item in sublist]
# # print(len(flat))
# d0 = [item for sublist in flat[0] for item in sublist]
# # print(d0)
# # data = pd.DataFrame(data)
# # print(data)

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None :
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False):
    if classifier_name=='fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  classifiers import  mlp
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='resnet':
        from  classifiers import resnet
        return resnet.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcnn':
        from  classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory,verbose)
    if classifier_name=='tlenet':
        from  classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory,verbose)
    if classifier_name=='twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory,verbose)
    if classifier_name=='encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='cnn': # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    if archive_name == 'mts_archive':
        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    else:
        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'+dataset_name
        x_train, y_train = read(file_name+'_TRAIN.txt')
        x_test, y_test = read(file_name+'_TEST.txt')
        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                                       y_test.copy())
    return datasets_dict


def create_directory(directory_path):
    if os.path.exists(directory_path):
        print('Result Folder Already Exist!!')
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            print('Something wrong to make folder??')
            return None
        return directory_path


classifier_name = 'mlp'
analysis_name = 'univariate'
dataset_name = 'mlp_L_3_1_70_'
root_dir = os.getcwd()
output_directory = 'D:/python/dl-4-tsc/arfftocsv-master/arfftocsv-master/results/'+classifier_name+'/'+archive_name+itr+'/'+\
                   dataset_name+'/'
print('output_directory', output_directory)

# output_directory = create_directory(output_directory)
# output_directory = 'D:/python/dl-4-tsc/arfftocsv-master/arfftocsv-master/results/fcn/ucr_itr_1/'+dataset_name+'/'

print('Method: ', archive_name, dataset_name, classifier_name, itr)

#
# def read(filename):
#     data = np.loadtxt(filename)  # delimiter=',')
#     Y = data[:, 0]
#     X = data[:, 1:]
#     return X, Y


datasets_dict = {}

# x_train, y_train = read('Coffee_TRAIN.txt')
# x_test, y_test = read('Coffee_TEST.txt')
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

print('output_directory', output_directory)
# classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
# classifier.fit(x_train, y_train, x_test, y_test, y_true)

# print(y_train)
# print(nb_classes)
# print(input_shape)
#
# print(os.getcwd())
