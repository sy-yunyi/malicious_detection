import pdb
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import detectModel
from Dutils import *
class MASQ:
    def __init__(self):
        super().__init__()
    def load_file(self,data_file,labels_file):
        """
        数据加载，masquerade的数据集分为两部分，分别进行加载
        数据部分，每个文件包含15000个指令，前5000个正常指令，后10000个为含有伪造指令的序列
        标签部分，给出了每个文件后10000指令序列中是否含有伪造指令，所以这里要补足，前5000个正常指令的标签
        返回：
            data 返回二维数组，(7500,100)
            labels 返回一维数组
        """
        with open(labels_file,'r') as fp:
            lines = [line.strip().split() for line in fp.readlines()]
            labels = np.array(lines).astype("int32")
        file_path = glob(data_file)
        data = []
        for file in file_path:
            with open(file,'r') as fp:
                lines = [line.strip() for line in fp.readlines()]
                data.append(lines)
        nor_labels = np.array([0]*2500).reshape(50,50)
        labels = np.hstack((nor_labels,labels.T))
        data = np.array(data).reshape((7500,100))
        return data,labels.flatten()
    def trian_model(self,data,labels,input_dim,st,iss_dim=1,model_type="SEN"):
        """
        模型训练
        data，为全部数据，格式为二维数组
        labels为数据标签
        input_dim
        st
        iss_dim 问题维度，多分类或者二分类
        model_type:语义模型，序列模型
        """
        data = data.reshape((data.shape[0],st,input_dim))
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
        if model_type=="SEN":
            model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st,iss_dim=iss_dim,act="tanh")
            model_pre = model.predict_proba(data)
            score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
            print(score)
            pre_c = model.predict_classes(X_test)
            y_test = y_test.astype("int32")
            print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))
            print(classification_report(y_test, pre_c))
        elif model_type=="SQU":
            model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st,iss_dim=iss_dim,act="tanh")
            model_pre = model.predict_proba(data)
            print(model.evaluate(X_test, y_test,batch_size=256, verbose=1))
            pre_c = model.predict_classes(X_test)
            # 二分类问题
            # y_test = y_test.astype("int32")
            # 多分类问题
            # y_test = y_test.argmax(axis=1)
            y_test = y_test.astype("int32") if iss_dim==1 else y_test.argmax(axis=1)
            print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))
            
        return model,model_pre

        


if __name__ == "__main__":
    data_file = "masquerade-data\*"
    labels_file = "masquerade_summary.txt"
    masq = MASQ()
    data,labels = masq.load_file(data_file,labels_file)

    _,data_vec = data2vec(data,type="doc2vec")

    input_dim = 5000
    st = 2
    # next_gram
    # data_next = squence_next_gram(data)
    # input_dim = 5959
    # st = 2
    
    masq.trian_model(data_vec,labels,input_dim,st)
    # data = data_next.reshape((data_next.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)

    # model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st)
    # score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    # print(score)
    # pre_c = model.predict_classes(X_test)
    # # # 二分类问题
    # y_test = y_test.astype("int32")
    # print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))

