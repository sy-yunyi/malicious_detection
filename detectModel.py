'''
@Descripttion: 
@version: 
@Author: Six
@Date: 2019-12-22 16:33:03
@LastEditors  : Six
@LastEditTime : 2019-12-29 17:14:10
'''
import tensorflow as tf
import keras
import numpy as np
import pdb
import sys
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from keras.models import load_model
import keras.metrics
import Dutils

def SemanticModel(X_train,y_train,X_test,y_test,data,input_dim,st=2,iss_dim=1,act = 'tanh'):
    """
    X_train : 训练数据
    y_train ： 数据标签
    data ： 全部数据
    input_dim ： 数据输入维度
    st ： 时间步长
    iss_dim ： 问题维度，二分类和多分类问题
    act: 激活函数
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-1))
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2,input_shape=(data.shape[0],st,input_dim),return_sequences=True))
    # ,return_sequences=True
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(keras.layers.Dense(iss_dim, activation=act))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=100, batch_size=512,verbose=1)
    return model 


def pay_xgboost(X_train=None,y_train=None,iss_dim =1):
    if iss_dim > 1:
        objective = "multi:softmax"  
        y_train = np.argmax(y_train,axis=1)
    else:
        objective = 'binary:logistic'
    
    clf = XGBClassifier(
    n_estimators=30,
    learning_rate =0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= objective,
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
    return clf.fit(X_train,y_train)

def EmbedingModel(labels,extra_fea=None,iss_dim=1,**model):
    """
    
    labels : data对应的标签
    extra_fea ： 针对数据提取的额外特征
    **model ： 不同模型的预测结果
    """
    
    embeding_data = np.hstack(([model[key] for key in model.keys()]))
    if extra_fea:
        embeding_data = np.hstack((extra_fea,embeding_data))
    
    X_train, X_test, y_train, y_test = train_test_split(embeding_data, labels, test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(X_train, y_train)

    #模型评估
    y_submission = clf.predict_proba(X_test).argmax(axis=1)
    # y_submission = y_submission if iss_dim==1 else y_submission.argmax(axis=1)
    
    # pre_c = clf.predict_classes(X_test)
    # print("Linear stretch of predictions to [0,1]")
    # y_test = y_test.astype("int32") if iss_dim==0 else y_test.argmax(axis=1)
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("val auc Score: %f" % (accuracy_score(y_test, y_submission)))
    print(classification_report(y_test, y_submission))
    pre_c = clf.predict(X_test)

    # 二分类问题
    # y_test = y_test.astype("int32")
    # 多分类问题
    # y_test = y_test.argmax(axis=1)
    
    print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))
    print(f1_score(y_test, y_submission,average="weighted"),precision_score(y_test, y_submission,average="weighted"),accuracy_score(y_test, y_submission),recall_score(y_test, y_submission,average="weighted"))
    Dutils.plot_confusion_matrix(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))),classes=range(2))
    
    # print(metrics.confusion_matrix(y_test.astype("int32"),pre_c.reshape((pre_c.shape[0]))))
    
def alg_model(data,labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    # knn = KNeighborsClassifier(X_train,y_train)
    # y_submission = knn.predict_proba(X_test).argmax(axis=1)
    # print("knn val auc Score: %f" % (accuracy_score(y_test, y_submission)))

    clf_rbf = svm.SVC(kernel='rbf')
    clf_rbf.fit(X_train,y_train)
    score_rbf = clf_rbf.score(X_test,y_test)
    print("svm score of rbf is : %f" % score_rbf)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    predictions = lg.predict(X_test)
    print("lg val auc Score: %f" % (accuracy_score(y_test, predictions)))



    
    
    
