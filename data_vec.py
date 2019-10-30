import os
import re
import urllib
import gensim
import pdb
import numpy as np
import tf_glove
import tensorflow as tf
from sklearn import preprocessing
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
import string
from sklearn.externals import joblib
def dataExtract(file_path,out_path):
    header = ["User-Agent:","Pragma:","Cache-control:","Accept:","Accept-Encoding:","Accept-Charset:","Accept-Language:","Host:","Cookie:","Connection:","Content-Length:","Content-Type:","Set-cookie"]
    data_list = []
    with open(file_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line != "\n" and line.split()[0] not in header and not line.startswith("Accept"):
                data_list.append(line)

    with open(out_path,"w") as pf:
        for d in data_list:
            pf.write(d)


def data2sen_e(file_paths,batch):
    data_set_l = []
    end_path_set_l = []
    arg_name_set_l = []
    arg_val_set_l = []
    labels = []
    for i in range(len(file_paths)):
        data_set,end_path_set,arg_name_set,arg_val_set = data2sen(file_paths[i])
        data_set_l.extend(data_set)
        end_path_set_l.extend(end_path_set)
        arg_name_set_l.extend(arg_name_set)
        arg_val_set_l.extend(arg_val_set)
        labels.extend([i] * (len(data_set)//batch))
    return data_set_l,end_path_set_l,arg_name_set_l,arg_val_set_l,labels



def data2sen(file_path):
    with open(file_path,'r') as fp:
        data_g = (d for d in fp.readlines())
        # data_e = []
        data_set = []
        end_path_set = []
        arg_name_set = []
        arg_val_set = []
        for d in data_g:
            if d.split()[0] == "GET":
                # data_e.append([d.strip()])
                # d = urllib.parse.unquote(urllib.parse.unquote(d.strip()))
                d = d.strip().split()
                act = d[0] # 方法
                end_p = d[1].split("?")[0].split("/")[-1] # 最后路径
                end_path_set.append(end_p)
                if len(d[1].split("?")) == 1:
                    data_set.append(" ".join([act,end_p]))
                    arg_name_set.append(" ")
                    arg_val_set.append(" ")
                else:
                    args = d[1].split("?")[-1].replace("+","#")
                    args_sp = re.split("[=&]",args)
                    data_set.append(urllib.parse.unquote(" ".join([act,end_p," ".join(args_sp)])))
                    arg_name_set.append(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0])))
                    arg_val_set.append(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 !=0])))

                # data_set.append(" ".join([d[0]," ".join(re.split("[?=&]",d[1].split("/")[-1]))]))

            elif d.split()[0] == "POST" or d.split()[0] == "PUT":
                # data_e.append([d.strip(),data_g.__next__().strip()])

                ac_end = re.split("[ /]",d.strip()) # 0:post,-3:end
                end_path_set.append(ac_end[-3])
                # args = urllib.parse.unquote(urllib.parse.unquote(data_g.__next__().strip())).replace("+"," ")
                args = data_g.__next__().strip().replace("+","#")
                
                args_sp = re.split("[=&]",args)

                arg_name_set.append(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0])))
                arg_val_set.append(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 !=0])))

                data_set.append(urllib.parse.unquote(" ".join([ac_end[0],ac_end[-3]," ".join(re.split("[=&]",args))])))
    return data_set,end_path_set,arg_name_set,arg_val_set
        

def word2vec_g(data,train_size):
    sentences = gensim.models.word2vec.LineSentence(data,max_sentence_length=10000)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences,hs=1,min_count=1,window=3,size=train_size)
    return model

def doc2vec_g(data):
    print("训练模型...")
    # mords为分割后的单词列表
    docm = [gensim.models.doc2vec.TaggedDocument(words = words.split(),tags = [i]) for i,words in enumerate(data)]

    model = gensim.models.doc2vec.Doc2Vec(documents=docm,dm=1,vector_size=100,window=8,min_count=5,workers=4)

    return model

    # np.savetxt('D:\six\code\malicious_detection\data\d2v.txt',model.docvecs)
    #单词向量 ： model.wv.vectors; 单词：model.wv.vocab
    #句子向量 ：model.docvecs.vectors_docs
    
def golvec_tf(data):
    print("训练模型...")
    model = tf_glove.GloVeModel(embedding_size=50, context_size=10, min_occurrences=25,learning_rate=0.05, batch_size=512)
    model.fit_to_corpus([words.split() for words in data])
    model.train(num_epochs=50, log_dir="malicious_detection/log/example", summary_batch_interval=1000)
    pdb.set_trace()
    

# 对末端路径和参数名进行one-hot编码
def one_hot(end_path_set,arg_name_set):
    # enc = preprocessing.OneHotEncoder()
    # end_path_np = np.array(end_path_set)
    # end_path_enc = enc.fit(end_path_np.reshape([end_path_np.shape[0],1]))

    enc1 = preprocessing.OneHotEncoder()
    arg_name_np = np.array([i for ans in arg_name_set for i in re.split("[ ]",ans)])
    arg_name_enc = enc1.fit(arg_name_np.reshape(arg_name_np.shape[0],1))

    return arg_name_enc

# 行为分析模型数据向量化
def beh_vec(data_set,end_path_enc,arg_name_enc,arg_val_vec,batch=50):
    # batch = 50
    be_vec = []
    for i in range(len(data_set)//batch):
        b_i = []
        b_data = data_set[i * batch: (i+1)*batch]
        for eb in b_data:
            eb = eb.split()
            b_i.append(0. if eb[0]=="GET" else 1.)
            try:
                b_i.extend(end_path_enc[eb[1]].tolist())
                arg_ns = [eb[2:][i] for i in range(len(eb[2:])) if i % 2 == 0]
                arg_vs = [eb[2:][i] for i in range(len(eb[2:])) if i % 2 != 0]
                for i in range(len(arg_ns)):
                    # 这里是one-hot,如果采用词袋，则将这里的值相加
                    b_i.extend(arg_name_enc.transform([[arg_ns[i]]]).toarray()[0])
                    b_i.extend(arg_val_vec[arg_vs[i]].tolist())
                # print(b_i)
            except:
                pass
        be_vec.append(b_i)
    return be_vec


def n_gram_pay(data,batch=50,n=2):
    n_gram_data = []
    for i in range(len(data)//batch):
        p_data = " ".join(data[i*batch:(i+1)*batch])
        n_gram_data.append(" ".join([p_data[j:j+n] for j in range(len(p_data)-(n+1))]))
    

    # 输出文件
    # with open("malicious_detection/data/{n}_gram_data.txt".format(n=n),"w",encoding='utf-8') as fp:
    #     fp.write("\n".join(n_gram_data))


def payload_pre(file_n_gram,arg_val_set,batch):
    pay_exr_fe = []
    for i in range(len(arg_val_set)//batch):
        v_data = " ".join(arg_val_set[i * batch: (i+1)*batch]).split()
        per_max_pay_l = max([len(e) for e in v_data])
        avg_pay_l = np.mean([len(e) for e in v_data])
        num_pay = len(v_data)
        re_ratio = len(set(v_data)) / num_pay
        pay_exr_fe.append([per_max_pay_l,avg_pay_l,num_pay,re_ratio])

    with open(file_n_gram,'r',encoding="utf-8") as fp:
        lines = fp.readlines()
        model = doc2vec_g([line.strip() for line in lines])

# 提取额外特征
def extra_feature(arg_name_set,end_path_set,arg_val_set,batch):

    # %3A 换行符
    # 还需要再进行解码
    pay_ext_fe = []
    for i in range(len(arg_val_set)//batch):

        # 最大频率参数比例
        n_data = " ".join(arg_name_set[i*batch:(i+1)*batch]).split()
        arg_counter = Counter(n_data)
        re_ration_name = arg_counter[max(arg_counter,key = arg_counter.get)] / len(n_data)

        # 末端路径重复率
        p_data = end_path_set[i*batch:(i+1)*batch]
        re_ration_path = len(p_data) / len(set(p_data))

        v_data = " ".join(arg_val_set[i * batch: (i+1)*batch]).split()    
        # 单个payload 最长长度
        per_max_pay_l = max([len(e) for e in v_data])
        # payload 长度均值
        avg_pay_l = np.mean([len(e) for e in v_data])
        # payload 长度方差
        std_pay_l = np.var([len(e) for e in v_data])
        # payload 重复率
        num_pay = len(v_data)
        
        re_ratio_pay = num_pay / len(set(v_data)) 
        v_s_data = " ".join(v_data)
        # 空格个数
        num_space = v_s_data.count("#")
        # 特殊字符：/@()%$-<>?
        num_ss = sum([v_s_data.count(i) for i in "/@()%$-<>?"])
        # 不可打印字符
        num_not_print = len([i for i in v_s_data if i not in string.printable])
        # 最大频率参数比例，末端路径重复率，单个payload最大长度，payload平均长度，payload长度方差，payload数量，payload重复率，空格个数，特殊字符数量，不可打印字符数量
        pay_ext_fe.append([re_ration_name,re_ration_path,per_max_pay_l,avg_pay_l,std_pay_l,num_pay,re_ratio_pay,num_space,num_ss,num_not_print])


if __name__ == "__main__":
    # 将目标数据才从源数据中提取出来
    # file_path = r"F:\广电\source\detection_web_attack\data\normalTrafficTest.txt"
    # out_path = r"malicious_detection\data\data_normalTrafficTest"

    file_path = r"F:\广电\source\detection_web_attack\data\anomalousTrafficTest.txt"
    out_path = r"malicious_detection\data\data_anomalousTrafficTest"
    dataExtract(file_path,out_path)

    #数据向量化
    # 1.数据提取
    # 正常流量
    # file_path = r"D:\six\code\malicious_detection\data\data_normalTrafficTest"
    # 异常流量
    # file_path = r"D:\six\code\malicious_detection\data\data_anomalousTrafficTest"
    
    # data_set,end_path_set,arg_name_set,arg_val_set = data2sen(file_path)
    # with open("malicious_detection/data/end_path_anomal.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(end_path_set))
    # with open("malicious_detection/data/arg_name_anomal.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(arg_name_set))
    # with open("malicious_detection/data/arg_val_anomal.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(arg_val_set))
    # with open("malicious_detection/data/data_all_anomal.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(data_set))

    # 2.向量化
    # 行为分析模型
    # 末端路径，参数名，payload，payload 最长长度

    file_paths = [r"D:\six\code\malicious_detection\data\data_normalTrafficTest",r"D:\six\code\malicious_detection\data\data_anomalousTrafficTest"]
    
    # # 合并arg_val,end_path文件
    # word_paths = ["arg_val.txt","arg_val_anomal.txt"]
    # with open("malicious_detection/data/arg_val_all.txt",'w',encoding='utf-8') as fp:
    #     data = []
    #     for i in range(len(word_paths)):
    #         with open ("malicious_detection/data/{wd}".format(wd=word_paths[i]),'r',encoding='utf-8') as fs:
    #             data.extend(fs.readlines())
    #     fp.write("".join(data))
    # end_paths = ["end_path.txt","end_path_anomal.txt"]
    # with open("malicious_detection/data/end_path_all.txt",'w',encoding='utf-8') as fp:
    #     data = []
    #     for i in range(len(end_paths)):
    #         with open ("malicious_detection/data/{wd}".format(wd=end_paths[i]),'r',encoding='utf-8') as fs:
    #             data.extend(fs.readlines())
    #     fp.write("".join(data))


    batch = 10
    n = 3
    data_vec = []
    data_set,end_path_set,arg_name_set,arg_val_set,labels = data2sen_e(file_paths,batch)
    np.save("labels",labels)
#     data_set,end_path_set,arg_name_set,arg_val_set = data2sen(file_paths[i])
#     # 行为语义学习模型
    arg_name_enc = one_hot(end_path_set,arg_name_set)
    arg_val_vec = word2vec_g(data = "malicious_detection/data/arg_val_all.txt",train_size = 100)
    end_path_enc = word2vec_g(data = "malicious_detection/data/end_path_all.txt",train_size = 20)
    bv = beh_vec(data_set,end_path_enc,arg_name_enc,arg_val_vec,batch)
    np.save("beh_vec_v100",bv,allow_pickle=True)
    # joblib.dump(bv, "beh_vec.pkl")
    pdb.set_trace()
    


    # 对payload进行n-gram划分，得到划分文件
    # n_gram_pay(arg_val_set,batch,n=2)

    # payload 表示模型
    # arg_val_vec = word2vec_g(data = "malicious_detection/data/{n}_gram_data.txt".format(n=n),train_size = 200)
    # file_n_gram = "malicious_detection/data/{n}_gram_data.txt".format(n=n)

    # payload_pre(file_n_gram,arg_val_set,batch)

    # extra_feature(arg_name_set,end_path_set,arg_val_set,batch)

    # pdb.set_trace()
    # doc2vec_g(data)
    # golvec_tf(data)


