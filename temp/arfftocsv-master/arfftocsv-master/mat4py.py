import scipy.io
import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.preprocessing import LabelEncoder

data = scipy.io.loadmat("Libras.mat")
data = data['mts']
flat = np.array(data).flatten()
print(len(flat))

flat = [item for sublist in flat for item in sublist]
# print(len(flat))
d0 = [item for sublist in flat[0] for item in sublist]
# print(d0)
# data = pd.DataFrame(data)
# print(data)

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
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


classifier_name = 'fcn'
archive_name = 'ucr'
itr = '_itr_1'
dataset_name = 'Coffee'
root_dir = os.getcwd()
output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+\
                   dataset_name+'/'
print('output_directory', output_directory)

# output_directory = create_directory(output_directory)
output_directory = 'D:/python/dl-4-tsc/arfftocsv-master/arfftocsv-master/results/fcn/ucr_itr_1/Coffee/'

print('Method: ', archive_name, dataset_name, classifier_name, itr)


def read(filename):
    data = np.loadtxt(filename)  # delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


datasets_dict = {}

x_train, y_train = read('Coffee_TRAIN.txt')
x_test, y_test = read('Coffee_TEST.txt')
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
classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
classifier.fit(x_train, y_train, x_test, y_test, y_true)

# print(y_train)
# print(nb_classes)
# print(input_shape)
#
# print(os.getcwd())
