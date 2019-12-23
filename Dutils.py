import itertools
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence,Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import tf_glove
import pdb
import numpy as np
from collections import Counter
import string
def groupBy(data=None,window=5,step=1):
    """
    序列分组
    data:[1,2,3,1,2,3]
    """
    i = 0
    data_group = []
    while(i+window <= len(data)):
        gd = data[i:(i+window)]
        i = i + step
        try:
            data_group.append(gd)
            # data_group.append(list(itertools.chain(*gd)))
        except:
            data_group.append(gd)
    return data_group

def groupBySqu(method_list,end_path_list,arg_name_list,arg_val_list,lable=0,window=5,step=5):
    """
    method_list,end_path_list,arg_name_list,arg_val_list: 数据，格式为[[1,2],[3,4],[5,6],[7,8]]
    划分目的为：[[1,2,3,4],[5,6,7,8]]
    window：每个划分的大小
    step：步长
    lable：数据标签
    返回值：
        data:划分后的数据，格式为[[1,2,3,4],[5,6,7,8]]
        labels: 标签列表，格式为[1,1]
        end_group : 每个序列中末端路径列表
        arg_name_group ：每个序列中参数名称列表
        arg_val_group ：每个列表中参数值列表
    """
    i=0
    data_group = []
    end_group = []
    labels = []
    arg_name_group = []
    arg_val_group = []
    while((i+window) < len(method_list)):
        gd = []
        method_g = method_list[i:i+window]
        end_g = end_path_list[i:i+window]
        arg_name_g = arg_name_list[i:i+window]
        arg_val_g = arg_val_list[i:i+window]
        i = i + step
        for j in range(window):
            args = []
            for n,v in zip(arg_name_g[j],arg_val_g[j]):
                args.append(n)
                args.append(v)
            gd.extend(method_g[j]+end_g[j]+args)
        data_group.append(gd)
        labels.append(lable)
        end_group.append(list(itertools.chain(*end_g)))
        arg_name_group.append(list(itertools.chain(*arg_name_g)))
        arg_val_group.append(list(itertools.chain(*arg_val_g)))
    return data_group,labels,end_group,arg_name_group,arg_val_group





def data2vec(data,vec_size = 100,type="doc2vec"):
    """
    data : 输入数据，格式为二维数组，[[1,2,3],[4,5,6]]
    vec_size : 词嵌入的维度
    type : 选择词嵌入类型，包括word2vec,doc2vec,glove
        word2vec 模型获得词向量model.wv[key]
        doc2vec  模型获得词向量model[key]；句向量 model.docvecs
        golve    模型获得词向量model.embedding_for(key)
    输出：
        输入正确的嵌入类型则输出嵌入模型和嵌入后的数据
        否则输出类型错误，返回0
    """
    print("进行词嵌入...")
    if type=="doc2vec":
        # words为分割后的单词列表
        docm = [TaggedDocument(words = words,tags = [i]) for i,words in enumerate(data)]
        # docm = [gensim.models.doc2vec.TaggedDocument(words = words.split(),tags = [i]) for i,words in enumerate(data)]
        model = Doc2Vec(documents=docm,dm=1,vector_size=vec_size,window=8,min_count=1,workers=4)
        data_vec = [model[di].flatten().tolist() for di in data]
    elif type=="word2vec":
        # sentences = LineSentence(data,max_sentence_length=10000)
        model = Word2Vec(sentences=data,hs=1,min_count=1,window=3,size=vec_size)
        data_vec = [model[di].flatten().tolist() for di in data]
    elif type=="glove":
        model = tf_glove.GloVeModel(embedding_size=vec_size, context_size=10, min_occurrences=1,learning_rate=0.05, batch_size=512)
        model.fit_to_corpus([words for words in data])
        model.train(num_epochs=50, log_dir="malicious_detection/log/example", summary_batch_interval=1000)
        data_vec = []
        for di in data:
            data_vec.append(np.array([model.embedding_for(div) for div in di]).flatten().tolist())
    else:
        print("Don't support the type")
        return 0,0
    return model,np.array(data_vec)
    

def squence_next_gram(data,min_count=2,max_count=2,analyzer = "word"):
    """
    data : 待统计数据，格式为[[1,2,3,4],[5,6,7,8]]
    min_count : ngram 最小跨度
    max_count : ngram 最大跨度
    """
    # CountVectorizer 的输入为["1 2 3 4","4 5 6 7"]
    data = [" ".join(d) for d in data]

    vectorizer = CountVectorizer(min_df=1, analyzer=analyzer, ngram_range=(min_count, max_count))
    vecmodel = vectorizer.fit(data)
    squence_n = vecmodel.transform(data).toarray()
    # pdb.set_trace()
    # (6105,32360)(6105,2484)
    return squence_n



def extra_squence_feature(arg_name_set,end_path_set,arg_val_set):
    '''
    @description: 提取序列特征
    @param {type} 
        arg_name_set
        end_path_set
        arg_val_set
    @return: 
        pay_ext_fe: 序列特征，格式为[[1,2,3],[4,5,6]]
    @author: Six
    @Date: 2019-12-23 18:45:47
    '''    

    # %0A 换行符
    # 还需要再进行解码
    pay_ext_fe = []
    for i in range(len(arg_val_set)):
        # 最大频率参数比例
        arg_counter = Counter(arg_name_set[i])
        re_ration_name = arg_counter[max(arg_counter,key = arg_counter.get)] / sum(arg_counter.values())

        # 末端路径重复率
        re_ration_path = len(end_path_set[i]) / len(set(end_path_set[i]))
  
        # 单个payload 最长长度
        per_max_pay_l = max([len(e) for e in arg_val_set[i]])
        # payload 长度均值
        avg_pay_l = np.mean([len(e) for e in arg_val_set[i]])
        # payload 长度方差
        std_pay_l = np.var([len(e) for e in arg_val_set[i]])
        # payload 重复率
        num_pay = len(arg_val_set[i])
        re_ratio_pay = num_pay / len(set(arg_val_set[i])) 

        v_s_data = " ".join(arg_val_set[i])
        # 空格个数
        num_space = v_s_data.count("^")
        # 特殊字符：/@()%$-<>?
        num_ss = sum([v_s_data.count(i) for i in "/@()%$-<>?"])
        # 不可打印字符
        num_not_print = len([i for i in v_s_data if i not in string.printable])
        # 最大频率参数比例，末端路径重复率，单个payload最大长度，payload平均长度，payload长度方差，payload数量，payload重复率，空格个数，特殊字符数量，不可打印字符数量
        pay_ext_fe.append([re_ration_name,re_ration_path,per_max_pay_l,avg_pay_l,std_pay_l,num_pay,re_ratio_pay,num_space,num_ss,num_not_print])

    return pay_ext_fe




def model2vec(model,data):
    data_vec = []
    for d in data:
        di_vec = [model[di].tolist() for di in d]
        data_vec.append(list(itertools.chain(*di_vec)))
    np.save("data_vec_new",data_vec,allow_pickle=True)
    
    pdb.set_trace()
        
    






if __name__ == "__main__":
    batch = 1
    file_paths = [r"D:\six\code\malicious_detection\data\data_normalTrafficTest",r"D:\six\code\malicious_detection\data\data_anomalousTrafficTest"]

    data,end_path_set,arg_name_set,arg_val_set,labels = data2sen_e(file_paths,batch)

    # data_file = r"D:\six\code\malicious_detection\data\demo.txt"
    # with open(data_file) as fp:
    #     data = fp.readlines()
    # data = [d.strip().split() for d in data]
    # d = groupBy(data,6,4)
    l = groupBy(labels,6,4)
    l = [min(li) for li in l]
    np.save("labels_new",l,allow_pickle=True)
    # m = doc2vec_g(d)
    # model2vec(m,d)


