import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os



def read_datasets(stock_name):
    # align timeline with '2005-07-19' (samsung has shortest data in FinanceDataReader)
    if stock_name == 'samsung':
        stock_data = fdr.DataReader('068270', '2005-07-19', '2019-09-01')
    elif stock_name == 'lg':
        stock_data = fdr.DataReader('066570', '2005-07-19', '2019-09-01')
    elif stock_name == 'naver':
        stock_data = fdr.DataReader('035420', '2005-07-19', '2019-09-01')
    elif stock_name == 'kakao':
        stock_data = fdr.DataReader('035720', '2005-07-19', '2019-09-01')
    elif stock_name == 'kospi200':
        stock_data = fdr.DataReader('KS200', '2005-07-19', '2019-09-01')
    return stock_data


def univariate_data_preprocessing(input_stock, column_name, data_length, gap_distance, percent_boundary):
    raw = input_stock.reset_index()[column_name].tolist()

    # univariate time series data for learnings
    data_2d = []
    for i in range(len(input_stock)-data_length-6):
        data_2d.append(raw[0+i: data_length+i])

    # univariate time series data with gap_distance for learnings
    data_divided = []
    for i in range(int((len(input_stock)-data_length-6)/gap_distance)):
        data_divided.append(raw[0+i*gap_distance : data_length+i*gap_distance])

    # need to changing normalizing step various method
    # normalizing with last price for each data
    normalized = []
    for data in data_divided:
        normalized.append([a/data[-1] for a in data])
    normalized = pd.DataFrame(normalized)

    # increase or decrease labeling each univariate time series data
    daylater_price = []
    for i in range(int(len(data_2d)/gap_distance)):
        k = i*gap_distance
        daylater_price.append((raw[data_length+k]-data_2d[k][-1]) / data_2d[k][-1] * 100)

    distrib_table = pd.DataFrame(columns=['Increase','Decrease','Drift sideway'], index=[0])
    distrib_table.loc[:,:] = 0

    label = []
    for percent in daylater_price:
        if abs(percent) < percent_boundary:
            distrib_table.loc[0,'Drift sideway'] += 1
            label.append(2)
        elif percent > percent_boundary:
            distrib_table.loc[0,'Increase'] += 1
            label.append(0)
        elif percent < percent_boundary:
            distrib_table.loc[0,'Decrease'] += 1
            label.append(1)
    print('distrib_table for percent_boundary(', percent_boundary, ')')
    print(distrib_table)

    label = pd.DataFrame(label)
    final_data = pd.concat([label, normalized], axis=1)

    return final_data


def create_directory(directory_path):
    if os.path.exists(directory_path):
        print('Result Folder Already Exist!!')
        return None
    else:
        try:
            os.makedirs(directory_path)
            print('success create result folder here \n', directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            print('Something wrong to make folder??')
            return None
        return directory_path


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
