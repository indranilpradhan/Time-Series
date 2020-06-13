import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import sys

def form_dataset(dataset,window_size):
    new_dataset = []
    new_label = []
    length = len(dataset)
    for i in range(length):
        last_index = i + window_size
        if(last_index > length-1):
            break
        temp_x = dataset[i:last_index]
        temp_y = dataset[last_index]
        new_dataset.append(temp_x)
        new_label.append(temp_y)
    return new_dataset, new_label

def fill_firstwindow(dataset,test_index,window_size):
    pred_first_window = []
    if(test_index[0] == 0):
        dataset = dataset.remove(0)
    cp_index = test_index.copy()
    for i in cp_index:
        if(i >= window_size):
            break
        test_index.remove(i)
        mean = np.mean(np.array(dataset[:i]))
        pred_first_window.append(mean)
        dataset[i] = mean
    return dataset, pred_first_window, test_index

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    final_result = []
    whole_df = pd.read_csv(str(sys.argv[1]), sep=';',parse_dates={'datetime' : ['Date', 'Time']}, infer_datetime_format=True, na_values=['nan','?'])

    t_df = whole_df.filter(["datetime","Global_active_power"],axis =1)
    copydf = t_df.filter(["Global_active_power"],axis=1)
    #print("54")

    test_index = list(copydf['Global_active_power'].index[copydf['Global_active_power'].apply(np.isnan)])
    test_index.sort()

    window_size = 120
    df120_mlp = copydf.copy()
    #print("60")

    dataset120_mlp = list(df120_mlp["Global_active_power"])
    dataset120_mlp,final_result,test_index = fill_firstwindow(dataset120_mlp,test_index,window_size)
    copyset120_mlp = dataset120_mlp.copy()
    #print("65")
    pre_xtrain, pre_ytrain = form_dataset(dataset120_mlp,window_size)
    xtest120_mlp = []
    #print("67")
    for i in test_index:
        xtest120_mlp.append(pre_xtrain[i-window_size])
    df120_mlp = df120_mlp.dropna(subset=["Global_active_power"])
    dataset120_mlp = list(df120_mlp["Global_active_power"])
    #print("70")
    pre_xtrain120_mlp, pre_ytrain120_mlp = form_dataset(dataset120_mlp,window_size)
    xtrain120_mlp = pre_xtrain120_mlp[:]
    ytrain120_mlp = pre_ytrain120_mlp[:]
    xtrain120_mlp = np.array(xtrain120_mlp)
    ytrain120_mlp = np.array(ytrain120_mlp)

    #print("77")
    model120_mlp_2 = Sequential()
    model120_mlp_2.add(Dense(1000, activation='relu', input_dim = window_size))
    model120_mlp_2.add(Dense(500, activation='relu'))
    model120_mlp_2.add(Dense(200, activation ='sigmoid'))
    model120_mlp_2.add(Dense(1))
    optimizer = SGD(lr=0.005, momentum=0.7)
    model120_mlp_2.compile(optimizer=optimizer, loss='mse')
    #print("fit")
    model120_mlp_2.fit(xtrain120_mlp, ytrain120_mlp, epochs=20, batch_size = 64, verbose=1)

    for i in test_index:
        t_dataset = np.array(copyset120_mlp[i-window_size:i])
        temp1 = (t_dataset.reshape(1,window_size))
        temp2 = (model120_mlp_2.predict(temp1)).reshape(1)[0]
        copyset120_mlp[i] = temp2
        final_result.append(temp2)

    print("Predicted values ")
    for i in final_result:
        print(i)


if __name__ == "__main__":
    main()