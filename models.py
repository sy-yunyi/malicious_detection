import tensorflow as tf
import keras
import numpy as np
import pdb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def beh_LSTM():
    data = np.load("beh_vec_v100.npy",allow_pickle=True)
    labels = np.load("labels.npy",allow_pickle = True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-100)
    # input_dim = 5651
    # data (6106,2,5651)
    input_dim = 8601
    # data (6106,2,8601)
    data = data.reshape((data.shape[0],2,input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-100))
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2, input_shape=(data.shape[0],2,input_dim)))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy']) 
    model.fit(data, labels, epochs=50, batch_size=512,verbose=1)
    score = model.evaluate(data, labels,batch_size=256, verbose=1)
    print(score)

    # pdb.set_trace()

def pay_LSTM(X_train, X_test, y_train, y_test):
    # data = np.load("pay_pre_doc2vec_word.npy",allow_pickle=True)
    # labels = np.load("labels.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-100)
    # input_dim = 10446
    # pdb.set_trace()
    # input_dim = 7701
    # data (6106,4,7701)
    # pdb.set_trace()
    # data = data.reshape((data.shape[0],4,input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-100))
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2, input_shape=(data.shape[0],4,input_dim)))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # keras.optimizers.adam
    model.compile(loss='mean_squared_error', optimizer="adam",metrics=['accuracy']) 
    model.fit(X_train, y_train, epochs=50, batch_size=256,verbose=1)
    score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    print(score)

def pay_xgboost(X_train, X_test, y_train, y_test):
    clf = XGBClassifier(
    n_estimators=30,
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
    model_sk=clf.fit(X_train, y_train)
    y_sklearn= clf.predict_proba(X_test)[:,1]
    pdb.set_trace()
    print(metrics.roc_auc_score(y_test, y_sklearn))




if __name__ == "__main__":
    # beh_LSTM()
    data = np.load("pay_pre_doc2vec_glove.npy",allow_pickle=True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    d1 = np.load("pay_ext_f.npy",allow_pickle=True)
    data = np.hstack((d1,data))

    # input_dim = 7701
    # input_dim = 8466
    # data = data.reshape((data.shape[0],4,input_dim))
    labels = np.load("labels.npy",allow_pickle = True)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # pay_LSTM(X_train, X_test, y_train, y_test)
    # pay_xgboost(X_train, X_test, y_train, y_test)

    # data = np.load("pay_ext_f.npy",allow_pickle=True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    pay_xgboost(X_train, X_test, y_train, y_test)
