'''
@Descripttion: 
@version: 
@Author: Six
@Date: 2019-12-23 10:12:10
@LastEditors  : Six
@LastEditTime : 2019-12-23 20:27:06
'''
import pdb
import re
from urllib.parse import unquote
import Dutils
from sklearn.model_selection import train_test_split
import detectModel
from sklearn import metrics
from sklearn.metrics.classification import classification_report
import keras
import numpy as np
class CSIC:
    def __init__(self):
        super().__init__()
    
    def load_file(self,file_path):
        '''
        @description: 加载CSIC数据，进行处理
        @param {file_path} 
        @return: 
            method_list : http使用方法，GET,POST,PUT，格式为[["GET],["POST"]]
            end_path_list : HTTP请求的末端路径，格式为[["index.jsp"],["index.jsp"]]
            arg_name_list ： 请求中的参数名称，格式为[["name","pass"],["name","pass","id"]]
            arg_val_list ： 请求中的参数值，格式为[["xiao","123"],["loi","124","43"]]
        @author: Six
        @Date: 2019-12-23 17:04:08
        '''
        with open(file_path,'r') as fp:
            lines = [line.strip() for line in fp.readlines()]
        headers = ["User-Agent","Pragma","Cache-control","Accept","Accept-Encoding","Accept-Charset","Accept-Language","Host","Cookie","Connection","Content-Length","Content-Type","Set-cookie"]
        data_list = []
        for line in lines:
            if line !="" and line.split(":")[0] not in headers:
                data_list.append(line)
        
        data_iter = data_list.__iter__()
        method_list = []
        end_path_list = []
        arg_name_list = []
        arg_val_list = []

        for d in data_iter:
            if d.split()[0] == "GET":
                method_list.append(["GET"])
                d = d.split()
                end_path_list.append([d[1].split("?")[0].split("/")[-1]])
                if len(d[1].split("?")) == 1:
                    arg_name_list.append([""])
                    arg_val_list.append([""])
                else:
                    args = d[1].split("?")[-1].replace("+","^")  
                    args_sp = re.split("[=&]",args)
                    arg_names = unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0]))
                    arg_names = arg_names.replace("%0A","").replace("%0D","")
                    arg_name_list.append(arg_names.split())
                    arg_vals = unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==1]))
                    arg_vals = arg_vals.replace("%0A","").replace("%0D","")
                    arg_val_list.append(arg_vals.split())
            elif d.split()[0] in ["POST","PUT"]:
                d = re.split("[ /]",d)
                method_list.append([d[0]])
                end_path_list.append([d[-3]])
                args = data_iter.__next__().replace("+","^")
                args_sp = re.split("[=&]",args)
                arg_names = unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0]))
                arg_names = arg_names.replace("%0A","").replace("%0D","")
                arg_name_list.append(arg_names.split())
                arg_vals = unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==1]))
                arg_vals = arg_vals.replace("%0A","").replace("%0D","")
                arg_val_list.append(arg_vals.split())
        return method_list,end_path_list,arg_name_list,arg_val_list

    def train_model(self,data,labels,input_dim=None,st=None,iss_dim=1,model_type="SEN"):
        if input_dim:
            data = data.reshape((data.shape[0],st,input_dim))
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
        if model_type=="SEN":
            model = detectModel.SemanticModel(X_train,y_train,data,input_dim,st,iss_dim=iss_dim,act="relu")
            model_pre = model.predict_proba(data)
            score = model.evaluate(X_test, y_test,batch_size=256, verbose=1)
            print(score)
            pre_c = model.predict_classes(X_test)
            y_test = y_test.astype("int32")
            print(metrics.confusion_matrix(y_test,pre_c.reshape((pre_c.shape[0]))))
            print(classification_report(y_test, pre_c))
        elif model_type=="PAY":
            model = detectModel.pay_xgboost(X_train,y_train)
            model_pre = model.predict_proba(data)
            pre_c = model.predict_proba(X_test).argmax(axis=1)
            print("auc score :"+str(metrics.roc_auc_score(y_test, pre_c)))
            print(metrics.confusion_matrix(y_test,pre_c))
            
        return model,model_pre

            


if __name__ == "__main__":
    csic = CSIC()

    file_path = "normalTrafficTraining.txt"
    method_list,end_path_list,arg_name_list,arg_val_list = csic.load_file(file_path)
    data,labels,end_group,arg_name_group,arg_val_group = Dutils.groupBySqu(method_list,end_path_list,arg_name_list,arg_val_list,lable=0,window=10,step=10)

    attack_file = "anomalousTrafficTest.txt"
    a_method_list,a_end_path_list,a_arg_name_list,a_arg_val_list = csic.load_file(attack_file)
    a_data,a_labels,a_end_group,a_arg_name_group,a_arg_val_group = Dutils.groupBySqu(a_method_list,a_end_path_list,a_arg_name_list,a_arg_val_list,lable=1,window=10,step=10)

    # 序列特征
    end_group = end_group + a_end_group
    arg_name_group = arg_name_group + a_arg_name_group
    arg_val_group = arg_val_group + a_arg_val_group

    exr_fea = Dutils.extra_squence_feature(arg_name_group,end_group,arg_val_group)
    
    # payload 表示 - ngram
    payload_pre = Dutils.squence_next_gram(arg_val_group,max_count = 2,analyzer = "char")

    
    
    data = data+a_data
    labels = np.array(labels + a_labels)

    _,pay_model = csic.train_model(payload_pre,labels,model_type="PAY")
    
    _,data_vec = Dutils.data2vec(data)

    data = keras.preprocessing.sequence.pad_sequences(data_vec,maxlen=max([len(i) for i in data_vec]),value=-10)
    data = np.hstack((data,exr_fea))
    input_dim = 12805
    st = 2
    _,squ_model = csic.train_model(data,labels,input_dim,st,iss_dim=1,model_type="SEN")

    detectModel.EmbedingModel(labels,extra_fea=exr_fea,pay_model = pay_model,squ_model=squ_model)
    pdb.set_trace()