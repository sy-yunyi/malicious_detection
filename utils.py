import itertools
from data_vec import doc2vec_g,data2sen_e
import pdb
import numpy as np
def groupBy(data=None,window=5,step=1):
    i = 0
    data_group = []
    while(i+window <= len(data)):
        gd = data[i:(i+window)]
        i = i + step
        try:
            data_group.append(list(itertools.chain(*gd)))
        except:
            data_group.append(gd)
    return data_group


def model2vec(model,data):
    data_vec = []
    for d in data:
        di_vec = [m[di].tolist() for di in d]
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


