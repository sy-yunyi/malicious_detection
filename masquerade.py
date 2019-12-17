import os
import glob 
import numpy as np
import itertools
import pdb
from data_vec import doc2vec_g
import keras
from models import beh_LSTM
from sklearn.model_selection import train_test_split

from masqierade_static import sequenceStatis,combained_


def load_file(file_path,label_path):

    with open(label_path,'r') as fp:
        lines = [i.strip().split() for i in fp.readlines()]
        lines = list(itertools.chain(*lines))
        np_lines = np.array(lines).reshape((100,50))
        labels = np_lines.T.reshape((1,5000))

    file = glob.glob(file_path)
    data = []
    train_data = []
    test_data = []
    for f in file:
        with open(f,'r') as fp:
            lines = [i.strip() for i in fp.readlines()]
            data.extend(lines)
            train_data.extend(lines[:5000])
            test_data.extend(lines[5000:])
    return data,train_data,test_data,labels

def data2vec(model,data,shape):
    print("data2vec....")
    datavec = []
    tmp = []

    for d in range(len(model.docvecs)):
        tmp.append(model.docvecs[d])
    np.save("masq_squence",tmp)
    pdb.set_trace()
    for d in data:
        datavec.extend(model[d].tolist())
    return np.array(datavec).reshape(shape)
    
def my_LSTM(X_train=None, X_test=None, y_train=None, y_test=None):
    print("LSTM")
    X_train = X_train.reshape((X_train.shape[0],100,100))
    X_test = X_test.reshape((X_test.shape[0],100,100))
    model = keras.models.Sequential()
    # pdb.set_trace()
    # model.add(keras.layers.LSTM(100))
    model.add(keras.layers.Masking(mask_value=-100))
    model.add(keras.layers.LSTM(50,batch_input_shape=(7500,X_train.shape[1], X_train.shape[2])))
    # model.add(Dense(1, activation='linear'))
    # LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector))

    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    # return model 
    model.fit(X_train, y_train, epochs=40, batch_size=128,verbose=1)
    score = model.evaluate(X_test, y_test,batch_size=128, verbose=1)
    print(score)

if __name__ == "__main__":
    file_path = "D:\six\code\masquerade-data\*"
    label_path = "D:\six\code\masquerade_summary.txt"
    data,train_data,test_data,labels = load_file(file_path,label_path)
    # sequenceStatis(data,True)
    # combained_(data)
    # pdb.set_trace()

    data = np.array(data).reshape((7500,100))
    model = doc2vec_g(data)
    # train data
    shape = (2500,10000)
    train_vec = data2vec(model,train_data,shape)
    train_labels= np.array([0] * 2500).reshape((2500,))
    # test data 
    shape = (5000,10000)
    test_vec = data2vec(model,test_data,shape)
    labels = labels.reshape(5000,)
    # my_LSTM(X_train=train_vec, X_test=test_vec, y_train=train_labels, y_test=labels)
    data = np.vstack((train_vec,test_vec))

    labels = np.hstack((train_labels,labels))

    # np.save("masq_data",data)
    # np.save("masq_labels",labels)
    # pdb.set_trace()
    # X_train, X_test, y_train, y_test = train_test_split(data, labels.T, test_size=0.2, random_state=0)

    # input_dim = 100
    # beh_LSTM(input_dim,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,data=data,st=100)


    

