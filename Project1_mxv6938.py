#!/usr/bin/env python
# coding: utf-8

#Importing the libraries
import pandas as pd
import numpy as np


def data_preprocessing():
    df = pd.read_csv("iris.csv")

    target_dict = {}
    for i,data in enumerate(df['species'].unique()):
        target_dict[data] = i

    df['target'] = df.apply(lambda x : target_dict[x[4]],axis=1)
    return df

def data_shuffling(df):
    q = df.values
    np.random.shuffle(q)
    df = pd.DataFrame(q)

    X = df.iloc[:, :-2].values
    Y = df.iloc[:, 5].values
    return df,X,Y

def data_split(df,X,Y):
    total_records = len(X)
    split_ratio = 80
    train_len = int(total_records * split_ratio / 100)

    X_train = X[0:train_len]
    Y_train = Y[0:train_len]
    X_test = X[train_len:total_records]
    Y_test = Y[train_len:total_records]
    return X_train,Y_train,X_test,Y_test


def init_parameters(lenw):
    w = np.random.rand(1,lenw)[0]
    b = 0
    return w, b


def forward_prop(X, w, b):
    y_hat = np.dot(w,X) + b
    return y_hat


def cost_function(y_hat,y):
    m = 1
    J = (1/(2*m)) * np.sum(np.square(y_hat-y))
    return J


def back_prop(X,y,y_hat):
    m = 1
    dy_hat = (1/m) * (y_hat-y)
    dw = np.dot(dy_hat, X.T)
    db = np.sum(dy_hat)
    return dw, db 


def gradient_desent(w,b,dw,db,lr):
    w = w - (lr * dw)
    b = b - (lr * db)
    return w,b


def train(X_train,Y_train):
    w,b = init_parameters(X_train.shape[1])
    cost_history = {}
    epochs = 10
    for j in range(epochs): 
        i=0
        for x_feed,y_feed in zip(X_train,Y_train):
            y_hat = forward_prop(x_feed,w,b)
            cost = cost_function(y_hat,y_feed)
            i+=1
            dw, db = back_prop(x_feed,y_feed,y_hat)
            w,b = gradient_desent(w,b,dw,db,1e-3)
        if j % 2 == 0:
            print("Epoch number - {} : Loop no - {} : Cost : {}".format(j,i,cost))
    print("Training constants \n w:{} b:{}".format(w, b))
    return w,b


def predict(X_test,Y_test, w, b):
    count = len(X_test)
    hit = 0
    for x_feed, y_feed in zip(X_test,Y_test):
        y_pred = np.round(forward_prop(x_feed,w,b))
        print("Predicted Value - {} : Actual Value - {}".format(y_pred, y_feed))
        if y_pred == y_feed:
            hit +=1
    return hit / count * 100



#To sample A fold for dataset
def data_sampler(X,Y, i,fold):
    length_data = len(X)
    partations = length_data / fold
    if not length_data % fold == 0:
        assert "Length of data = {} folds must be divisable".format(length_data)
    test_start = int(i * partations)
    test_stop = int((i+1) * partations)
    X_test = X[test_start:test_stop]
    Y_test = Y[test_start:test_stop]
    X_train = np.concatenate((X[:test_start],X[(test_stop):]))
    Y_train = np.concatenate((Y[:test_start],Y[(test_stop):]))

    return X_train, Y_train, X_test, Y_test


def cross_validation(X,Y):
    fold = 5
    ACC_dict = {}
    for i in range(5):
        X_train,Y_train,X_test,Y_test = data_sampler(X,Y,i,fold)
        w,b = train(X_train,Y_train)
        acc = predict(X_test,Y_test,w,b)
        ACC_dict[i] = acc
    print(ACC_dict)


def main():
    df = data_preprocessing()
    df,X,Y = data_shuffling(df)
    X_train, Y_train, X_test, Y_test = data_split(df,X,Y)
    print("Training and predicting results for the test data using the trained model:")
    w, b = train(X_train, Y_train)
    acc = predict(X_test, Y_test, w, b)
    print("Accuracy = {}%".format(acc))

    print("\n \nCross validation for the trained model")
    cross_validation(X,Y)

if __name__== "__main__":
  main()