from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import detectModel
from sklearn import metrics
import numpy as np

from Dutils import *
import pdb

class ADFA:
    """
    数据加载，输出为统一的格式。
    load_adfa_type_files()
        输入：
            文件路径格式为：例如-K:\Guang\code\Attack_Data_Master\*\*
            attack_tyep：是否区分攻击类型，默认为None，不区分则为二分类
            window：数据划分控制窗口，默认为None，不划分
            step：数据划分控制步长，默认为None，不划分
        输出：
            数据：经过划分的数据，划分采用函数groupBy实现。格式为二维数组，[[1,2,3,4],[5,6,7,8]]
            标签：数据对应的标签，格式为一维数组，[1,1,1,1,0,0,0,0,0]
    """
    def __init__(self):
        super().__init__()
    
    def load_type_file(self,filename,window,step):
        with open(filename,'r',encoding="windows-1252") as f:
            line = f.readline()
            line = line.strip('\n').split()
        if window:
            return groupBy(line,window,step),len(groupBy(line,window,step))
        else:
            return line,1

    def load_adfa_type_files(self,file_psth,att_type=None,window=None,step=None):
        x = []
        y = []
        allfile=glob(file_psth)
        
        for file in allfile:
            if att_type:
                for ty in attack_tyep.keys():
                    if ty in file:
                        data,lb = self.load_type_file(file,window,step)
                        if data ==[]:
                            continue
                        x.append(data) if type(data[0]) !=list else x.extend(data)
                        if ty == "Meter" and "Java_Meter" in file:
                            ty= "Java_Meter"
                        y.extend([attack_tyep[ty]] * lb) 
                        break
            else:
                if "Attack" not in file:
                    data,lb = self.load_type_file(file,window,step)
                    if data ==[]:
                        continue
                    x.append(data) if type(data[0]) !=list else x.extend(data)
                    y.extend([0] * lb)
                else:
                    data,lb = self.load_type_file(file,window,step)
                    if data ==[]:
                        continue
                    x.append(data) if type(data[0]) !=list else x.extend(data)
                    y.extend([1] * lb)
        return x, y
    def train_model(self,data,labels,input_dim=None,st=None,iss_dim=1,model_type="SEN"):
        """
        模型训练
        data，为全部数据，格式为二维数组
        labels为数据标签
        input_dim
        st
        iss_dim 问题维度，多分类或者二分类
        model_type:语义模型，序列模型
        """
        if input_dim:
            data = data.reshape((data.shape[0],st,input_dim))
        if iss_dim!=1:
                enc = OneHotEncoder()
                labels = np.array(labels).reshape(len(labels),-1)
                enc.fit(labels)
                labels = enc.transform(labels).toarray()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
        if model_type=="SEN":
            model = detectModel.SemanticModel(X_train,y_train,X_test,y_test,data,input_dim,st,iss_dim=iss_dim,act="tanh")
            model_pre = model.predict_proba(data)
            score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
            print(score)
            pre_c = model.predict_classes(X_test)
            # 二分类问题
            # y_test = y_test.astype("int32")
            # 多分类问题
            # y_test = y_test.argmax(axis=1)
            y_test = y_test.astype("int32") if iss_dim==1 else y_test.argmax(axis=1)
            print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))

        elif model_type=="SQU":
            model = detectModel.pay_xgboost(X_train,y_train,iss_dim)
            model_pre = model.predict_proba(data)
            pre_c = model.predict_proba(X_test).argmax(axis=1).reshape((-1,1))
            print("auc score :"+str(metrics.roc_auc_score(y_test, pre_c)))
            # y_test = y_test.argmax(axis=1).reshape((-1,1))
            # print(metrics.confusion_matrix(y_test,pre_c))

        return model,model_pre




if __name__ == "__main__":
    adfa = ADFA()
    attack_tyep = {"Adduser":1,"Hydra_FTP":2,"Hydra_SSH":2,"Java_Meter":4,"Meter":1,"Web":3}
    file_path = r".\Attack_Data_Master\*\*"

    window=100
    step=100

    attack_data,attack_y = adfa.load_adfa_type_files(file_path,window=window,step=step,att_type=attack_tyep)
    
    file_path = r".\Training_Data_Master\*"
    train_data,train_y = adfa.load_adfa_type_files(file_path,window=window,step=step)
    x = train_data + attack_data
    labels = np.array(train_y+attack_y)
    # pdb.set_trace()

    data_next = squence_next_gram(x)

    # X_train, X_test, y_train, y_test = train_test_split(data_next, labels, test_size=0.25, random_state=0)
    # model = detectModel.pay_xgboost(X_train,y_train).predict_proba(data_next)


    _,data_vec = data2vec(x,type="doc2vec")
    
    input_dim = data_next.shape[1]
    st = 1
    iss_dim = 5
    _,squ_model = adfa.train_model(data_next,labels,iss_dim=iss_dim,model_type="SQU")
    _,squ_model1 = adfa.train_model(data_next,labels,input_dim=input_dim,st=st,iss_dim=iss_dim)
    # data_vec = np.hstack((data_vec,data_next))
    input_dim = data_vec.shape[1]//2
    # input_dim = 3573
    st = 2
    _,sen_model = adfa.train_model(data_vec,labels,input_dim=input_dim,st=st,iss_dim=iss_dim)

    detectModel.EmbedingModel(labels,iss_dim=iss_dim,squ_model=squ_model,sen_model=sen_model)
    detectModel.EmbedingModel(labels,iss_dim=iss_dim,squ_model=squ_model,squ_model1=squ_model1)

    detectModel.EmbedingModel(labels,iss_dim=iss_dim,squ_model=squ_model,sen_model=sen_model,squ_model1=squ_model1)

    
    
    # data_next = squence_next_gram(x)
    # input_dim = 2146
    # st = 1
    # _,squ_model = adfa.trian_model(data_next,labels,input_dim,st,iss_dim=7)

    
    # data = data_vec.reshape((data_vec.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)

    # model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st)

    # sen_model = model.predict_proba(data)

    # score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
    # print(score)
    # pre_c = model.predict_classes(X_test)
    # # 二分类问题
    # y_test = y_test.astype("int32")
    # # 多分类问题
    # # y_test = y_test.argmax(axis=1)
    # print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))


    # data_next = squence_next_gram(x)

    # input_dim = 2146
    # st = 1
    # data = data_next.reshape((data_next.shape[0],st,input_dim))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)

    # model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st)

    # squ_model = model.predict_proba(data)

