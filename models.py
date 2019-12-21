import tensorflow as tf
import keras
import numpy as np
import pdb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import keras.metrics

from sklearn.externals import joblib

def beh_LSTM(input_dim,X_train=None, X_test=None, y_train=None, y_test=None,data=None,st=2):
    # data = np.load("beh_vec_v100.npy",allow_pickle=True)
    # labels = np.load("labels.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-100)
    # input_dim = 5651
    # data (6106,2,5651)
    # input_dim = 8601
    # data (6106,2,8601)
    # data = data.reshape((data.shape[0],2,input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-1))
    # pdb.set_trace()
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2,input_shape=(data.shape[0],st,input_dim),return_sequences=True))
    # ,return_sequences=True
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(keras.layers.Dense(6, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    # return model 
    model.fit(X_train, y_train, epochs=5, batch_size=512,verbose=1)
    score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    print(score)
    pre_c = model.predict_classes(X_test)
    # 二分类问题
    # y_test = y_test.astype("int32")
    # 多分类问题
    y_test = y_test.argmax(axis=1)
    print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))
    return model.predict_proba(data)
    

def pay_LSTM(input_dim,X_train=None, X_test=None, y_train=None, y_test=None,data=None):
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
    return model
    # model.fit(X_train, y_train, epochs=50, batch_size=256,verbose=1)
    # score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    # print(score)

def pay_xgboost(X_train=None, X_test=None, y_train=None, y_test=None):
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
    return clf
    # model_sk=clf.fit(X_train, y_train)
    # y_sklearn= model_sk.predict_proba(X_test)[:,1]
    # # pdb.set_trace()

    # print(metrics.roc_auc_score(y_test, y_sklearn))



def run_beh_LSTM():
    print("start beh_LSTM...")
    
    data = np.load("beh_vec_v100.npy",allow_pickle=True)
    labels = np.load("labels.npy",allow_pickle = True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    input_dim = 8601
    data = data.reshape((data.shape[0],2,input_dim))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # 训练模型
    # model = beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data)
    # model.fit(X_train, y_train, epochs=100, batch_size=512,verbose=1)
    # 加载模型
    model = load_model("beh_LSTM_model")
    score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)

    print("beh_LSTM score : "+str(score))
    # model.save("beh_LSTM_model")
    return model.predict_proba(data)

def run_pay_LSTM():
    print("start pay_LSTM...")
    
    data = np.load("pay_pre_doc2vec_glove.npy",allow_pickle=True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    d1 = np.load("pay_ext_f.npy",allow_pickle=True)
    data = np.hstack((d1,data))
    # # word2vec
    # input_dim = 15407
    # # glove
    input_dim = 16937
    data = data.reshape((data.shape[0],2,input_dim))
    labels = np.load("labels.npy",allow_pickle = True)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    #训练模型
    # model = pay_LSTM(input_dim,X_train, X_test, y_train, y_test,data)
    # model.fit(X_train, y_train, epochs=50, batch_size=256,verbose=1)

    # 加载模型
    model = load_model("pay_LSTM_model")
    score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    print("pay_LSTM score : "+str(score))
    # model.save("pay_LSTM_model_word")
    return model.predict_proba(data)

def run_pay_xgboost():
    print("start pay_xgboost...")
    data = np.load("pay_pre_doc2vec_word.npy",allow_pickle=True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    d1 = np.load("pay_ext_f.npy",allow_pickle=True)
    data = np.hstack((d1,data))
    labels = np.load("labels.npy",allow_pickle = True)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # 训练模型
    # clf = pay_xgboost(X_train, X_test, y_train, y_test)
    # model_sk=clf.fit(X_train, y_train)    
    # joblib.dump(model_sk, 'pay_xgboost_model.pkl')
    # 加载模型
    model_sk = joblib.load('pay_xgboost_model.pkl')

    
    y_sklearn= model_sk.predict_proba(X_test)[:,1]

    print("pay_xgboost score : "+str(metrics.roc_auc_score(y_test, y_sklearn)))
    return model_sk.predict_proba(data)



def blending_model():
    
    beh_pre = run_beh_LSTM()
    pay_pre = run_pay_LSTM()
    pay_xg_pre = run_pay_xgboost()
    exr = np.load("pay_ext_f.npy",allow_pickle=True)
    labels = np.load("labels.npy",allow_pickle = True)
    data = np.hstack((exr,beh_pre,pay_pre,pay_xg_pre))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    
    clf.fit(X_train, y_train)
    y_submission = clf.predict_proba(X_test)[:, 1]
    
    # pre_c = clf.predict_classes(X_test)
    # print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("val auc Score: %f" % (metrics.roc_auc_score(y_test, y_submission)))
    pre_c = clf.predict(X_test)
    print(metrics.confusion_matrix(y_test.astype("int32"),pre_c.reshape((pre_c.shape[0]))))




def stack_model():
    data_statis = np.load("masq_data_statis_only.npy",allow_pickle=True)
    labels = np.load("masq_labels.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-1)
    # pdb.set_trace()
    # input_dim = 8063
    # st = 3
    # input_dim = 14189
    # st = 1
    # data = data.reshape((data.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    # statis_model = beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)

    data = np.load("masq_data.npy",allow_pickle=True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-1)

    input_dim = 5000
    st = 2
    data = data.reshape((data.shape[0],st,input_dim))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    squence_model = beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)

    data = np.hstack((squence_model,data_statis))
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    
    clf.fit(X_train, y_train)
    y_submission = clf.predict_proba(X_test)[:, 1]
    # print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("val auc Score: %f" % (metrics.roc_auc_score(y_test, y_submission)))
    pre_c = clf.predict(X_test)
    print(metrics.confusion_matrix(y_test.astype("int32"),pre_c.reshape((pre_c.shape[0]))))



if __name__ == "__main__":
    
    # beh_LSTM
    # data = np.load("beh_vec_v100.npy",allow_pickle=True)
    # labels = np.load("labels.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    # input_dim = 8601
    # data = data.reshape((data.shape[0],2,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # beh_LSTM(input_dim,X_train, X_test, y_train, y_test)

    # pay_LSTM
    # data = np.load("pay_pre_doc2vec_glove.npy",allow_pickle=True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    # d1 = np.load("pay_ext_f.npy",allow_pickle=True)
    # data = np.hstack((d1,data))
    # # word2vec
    # # input_dim = 15407
    # # glove
    # input_dim = 16937
    # data = data.reshape((data.shape[0],2,input_dim))
    # labels = np.load("labels.npy",allow_pickle = True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # pay_LSTM(input_dim,X_train, X_test, y_train, y_test)
    
    #额外特征
    # data = np.load("pay_ext_f.npy",allow_pickle=True)
    # labels = np.load("labels.npy",allow_pickle = True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # pay_xgboost(X_train, X_test, y_train, y_test)
    
    # pay_xgboost    
    # data = np.load("pay_pre_doc2vec_word.npy",allow_pickle=True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    # d1 = np.load("pay_ext_f.npy",allow_pickle=True)
    # data = np.hstack((d1,data))
    # labels = np.load("labels.npy",allow_pickle = True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # pay_xgboost(X_train, X_test, y_train, y_test)

    
    # blending
    # blending_model()

    # run_beh_LSTM()
    # run_pay_LSTM()
    # run_pay_xgboost()

    # new model
    # data = np.load("data_vec_new.npy",allow_pickle=True)
    # labels = np.load("labels_new.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    # # pdb.set_trace()
    # input_dim = 1680
    # st = 10
    # data = data.reshape((data.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # # pdb.set_trace()
    # beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)

    # masq
    # data = np.load("masq_data_statis.npy",allow_pickle=True)
    # labels = np.load("masq_labels.npy",allow_pickle = True)
    # data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=1)
    # # pdb.set_trace()
    # input_dim = 8063
    # st = 3
    # # input_dim = 100
    # # st = 1
    # data = data.reshape((data.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    # # pdb.set_trace()
    # beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)
    
    # stack_model()
    
    # adfa

    data = np.load("adfa_data_type_50_next_gram.npy",allow_pickle=True)
    labels = np.load("adfa_labels_type.npy",allow_pickle = True)
    data = np.array([d.astype("float32") for d in data])
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-1.0)
    # pdb.set_trace()
    # input_dim = 8063
    # st = 3
    # input_dim = 1474  # or
    # input_dim = 2207  # or_statis
    # input_dim = 2926  # or_statis_50
    input_dim = 2146
    st = 1
    data = data.reshape((data.shape[0],st,input_dim))
    enc = OneHotEncoder()
    labels = np.array(labels).reshape(len(labels),-1)
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    # pdb.set_trace()
    next_g_data = beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)

    data = np.load("adfa_data_type_50_vec.npy",allow_pickle=True)
    # labels = np.load("adfa_labels_type.npy",allow_pickle = True)
    data = np.array([d.astype("float32") for d in data])
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-1.0)
    # pdb.set_trace()
    # input_dim = 8063
    # st = 3
    # input_dim = 1474  # or
    # input_dim = 2207  # or_statis
    # input_dim = 2926  # or_statis_50
    input_dim = 2500
    st = 2
    data = data.reshape((data.shape[0],st,input_dim))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    # pdb.set_trace()
    vec_data = beh_LSTM(input_dim,X_train, X_test, y_train, y_test,data,st)

    data = np.hstack((vec_data,next_g_data))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    y_train = y_train.argmax(axis=1)
    clf.fit(X_train, y_train)
    # y_submission = clf.predict_proba(X_test)[:, 1]
    y_submission = clf.predict_proba(X_test)
    
    # print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    # print("val auc Score: %f" % (metrics.roc_auc_score(y_test, y_submission)))
    print(metrics.roc_auc_score(y_test, y_submission, average='micro'))
    pre_c = clf.predict(X_test)

    # 二分类问题
    # y_test = y_test.astype("int32")
    # 多分类问题
    y_test = y_test.argmax(axis=1)
    print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))


