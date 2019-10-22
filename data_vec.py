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
def dataExtract(file_path,out_path):
    header = ["User-Agent:","Pragma:","Cache-control:","Accept:","Accept-Encoding:","Accept-Charset:","Accept-Language:","Host:","Cookie:","Connection:","Content-Length:","Content-Type:"]
    data_list = []
    with open(file_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line != "\n" and line.split()[0] not in header and not line.startswith("Accept"):
                data_list.append(line)

    with open(out_path,"w") as pf:
        for d in data_list:
            pf.write(d)

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
                    data_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([act,end_p," ".join(args_sp)]))))
                    arg_name_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0]))))
                    arg_val_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 !=0]))))

                # data_set.append(" ".join([d[0]," ".join(re.split("[?=&]",d[1].split("/")[-1]))]))

            elif d.split()[0] == "POST" or d.split()[0] == "PUT":
                # data_e.append([d.strip(),data_g.__next__().strip()])

                ac_end = re.split("[ /]",d.strip()) # 0:post,-3:end
                end_path_set.append(ac_end[-3])
                # args = urllib.parse.unquote(urllib.parse.unquote(data_g.__next__().strip())).replace("+"," ")
                args = data_g.__next__().strip().replace("+","#")
                
                args_sp = re.split("[=&]",args)

                arg_name_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 ==0]))))
                arg_val_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([a for i, a in enumerate(args_sp) if i % 2 !=0]))))

                data_set.append(urllib.parse.unquote(urllib.parse.unquote(" ".join([ac_end[0],ac_end[-3]," ".join(re.split("[=&]",args))]))))
    return data_set,end_path_set,arg_name_set,arg_val_set
        

def word2vec_g(data,train_size):
    sentences = gensim.models.word2vec.LineSentence(data,max_sentence_length=40000)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences,hs=1,min_count=1,window=3,size=train_size)
    return model

def doc2vec_g(data):
    print("训练模型...")
    # mords为分割后的单词列表
    docm = [gensim.models.doc2vec.TaggedDocument(words = words.split(),tags = [i]) for i,words in enumerate(data)]

    model = gensim.models.doc2vec.Doc2Vec(documents=docm,dm=1,vector_size=100,window=8,min_count=5,workers=4)

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
    enc = preprocessing.OneHotEncoder()
    end_path_np = np.array(end_path_set)
    end_path_enc = enc.fit(end_path_np.reshape([end_path_np.shape[0],1]))

    enc1 = preprocessing.OneHotEncoder()
    arg_name_np = np.array([i for ans in arg_name_set for i in re.split("[ ]",ans)])
    arg_name_enc = enc1.fit(arg_name_np.reshape(arg_name_np.shape[0],1))

    return end_path_enc,arg_name_enc

# 行为分析模型数据向量化
def beh_vec(data_set,end_path_enc,arg_name_enc,arg_val_vec):
    batch = 50
    be_vec = []
    for i in range(len(data_set)//batch):
        b_i = []
        b_data = data_set[i * batch: (i+1)*batch]
        for eb in b_data:
            eb = eb.split()
            b_i.append(0. if eb[0]=="GET" else 1.)
            b_i.extend(end_path_enc.transform([[eb[1]]]).toarray()[0])
            arg_ns = [eb[2:][i] for i in range(len(eb[2:])) if i % 2 == 0]
            arg_vs = [eb[2:][i] for i in range(len(eb[2:])) if i % 2 != 0]

            b_i.extend(arg_name_enc.transform([[i]]).toarray()[0] for i in arg_ns)
            b_i.extend(arg_val_vec[i] for i in arg_vs)
            # pdb.set_trace()




    


                    



if __name__ == "__main__":
    # 将目标数据才从源数据中提取出来
    # file_path = r"F:\广电\source\detection_web_attack\data\normalTrafficTest.txt"
    # out_path = r"malicious_detection\data\data_normalTrafficTest"
    # dataExtract(file_path,out_path)

    #数据向量化
    # 1.数据提取
    # file_path = r"D:\six\code\malicious_detection\data\data_normalTrafficTest"
    # data_set,end_path_set,arg_name_set,arg_val_set = data2sen(file_path)
    # with open("malicious_detection/data/end_path.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(end_path_set))
    # with open("malicious_detection/data/arg_name.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(arg_name_set))
    # with open("malicious_detection/data/arg_val.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(arg_val_set))
    # with open("malicious_detection/data/data_all.txt","w",encoding='utf-8') as fp:
    #     fp.write("\n".join(data_set))

    # 2.向量化
    # 行为分析模型
    # 末端路径，参数名，payload，payload 最长长度
    file_path = r"D:\six\code\malicious_detection\data\data_normalTrafficTest"
    data_set,end_path_set,arg_name_set,arg_val_set = data2sen(file_path)
    end_path_enc,arg_name_enc = one_hot(end_path_set,arg_name_set)
    arg_val_vec = word2vec_g(data = "malicious_detection/data/arg_val.txt",train_size = 200)
    beh_vec(data_set,end_path_enc,arg_name_enc,arg_val_vec)

    # pdb.set_trace()
    # doc2vec_g(data)
    # golvec_tf(data)


