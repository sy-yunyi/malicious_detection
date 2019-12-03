import os
import glob 
import numpy as np
import itertools
import pdb
from data_vec import doc2vec_g
import keras

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
    for d in data:
        datavec.extend(model[d].tolist())
    return np.array(datavec).reshape(shape)
    
def my_LSTM(X_train=None, X_test=None, y_train=None, y_test=None):
    print("LSTM")
    # pdb.set_trace()
    X_train = X_train.reshape((X_train.shape[0],2,2500))
    X_test = X_test.reshape((X_test.shape[0],2,2500))
    model = keras.models.Sequential()
    pdb.set_trace()
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2, input_shape=(X_train.shape[0],2,2500)))

    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    # return model 
    model.fit(X_train, y_train, epochs=40, batch_size=1024,verbose=1)
    score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    print(score)

if __name__ == "__main__":
    file_path = "D:\six\code\masquerade-data\*"
    label_path = "D:\six\code\masquerade_summary.txt"
    data,train_data,test_data,labels = load_file(file_path,label_path)
    data = np.array(data).reshape((7500,100))
    model = doc2vec_g(data)
    # train data
    shape = (5000,5000)
    train_vec = data2vec(model,train_data,shape)
    train_labels= np.array([0] * 5000).reshape((5000,))
    # test data 
    shape = (10000,5000)
    test_vec = data2vec(model,test_data,shape)
    my_LSTM(X_train=train_vec, X_test=test_vec, y_train=train_labels, y_test=labels)


    

