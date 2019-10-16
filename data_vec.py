import os
import re
import urllib
import gensim
import pdb
import numpy as np
import tf_glove
import tensorflow as tf
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
        for d in data_g:
            if d.split()[0] == "GET":
                # data_e.append([d.strip()])
                d = urllib.parse.unquote(urllib.parse.unquote(d.strip()))
                d = d.split()
                
                data_set.append(" ".join([d[0]," ".join(re.split("[?=&]",d[1].split("/")[-1]))]))

            elif d.split()[0] == "POST" or d.split()[0] == "PUT":
                # data_e.append([d.strip(),data_g.__next__().strip()])
                ac_end = re.split("[ /]",d.strip()) # 0:post,-3:end
                args = urllib.parse.unquote(urllib.parse.unquote(data_g.__next__().strip()))
                data_set.append(" ".join([ac_end[0],ac_end[-3]," ".join(re.split("[?=&]",args))]))
    return data_set
        
            
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
    model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)
    pdb.set_trace()
    

                    



if __name__ == "__main__":
    # 将目标数据才从源数据中提取出来
    # file_path = r"F:\广电\source\detection_web_attack\data\normalTrafficTest.txt"
    # out_path = r"malicious_detection\data\data_normalTrafficTest"
    # dataExtract(file_path,out_path)

    #数据向量化
    file_path = r"D:\six\code\malicious_detection\data\data_normalTrafficTest"
    data = data2sen(file_path)
    pdb.set_trace()
    # doc2vec_g(data)
    # golvec_tf(data)


