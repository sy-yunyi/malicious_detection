from glob import glob
import pdb
from collections import Counter as cc
import json
import numpy as np
from utils import groupBy 
from data_vec import doc2vec_g

def file2data(file_path):
    file_path_train = "K:\Guang\code\Training_Data_Master\*"
    file_path = "K:\Guang\code\Attack_Data_Master\*\*"
    files = glob(file_path_train)
    data_list = []

    train_data =[]
    for f in files:
        with open(f,'r') as fp:
            line = fp.readline().strip()
            train_data.extend(line.split())
    
    train = groupBy(train_data,50,50)
    labels = [0] * len(train)
    data_list.extend(train)
    


    files = glob(file_path)
    attack_data = []
    for f in files:
        with open(f,'r') as fp:
            line = fp.readline().strip()
            attack_data.extend(line.split())
    attack = groupBy(attack_data,50,50)
    data_list.extend(attack)
    labels.extend([1]*len(attack))
    np.save("adfa_labels_50",labels)


    # sequenceStatis(data_list,out_file="adfa_statis_next_50.json")
    # sequenceStatis(data_list,out_file="adfa_statis_double_50.json",trim=True)


def sequenceStatis(data,out_file,trim=False):
    shell_dict_all={}
    for j,d in enumerate(data):
        shell_dict={}
        for i,di in enumerate(d):
            if di not in shell_dict.keys():
                if i+1<len(d):
                    if trim and i-1>=0:
                        shell_dict[di]=[[d[i-1],d[i+1]]]
                    elif not trim:
                        shell_dict[di]=[d[i+1]]
            else:
                if i+1<len(d):
                    if trim and i-1>=0:
                        shell_dict[di].append([d[i-1],d[i+1]])
                    elif not trim:
                        shell_dict[di].append(d[i+1])
        shell_dict_all["user"+str(j)]=shell_dict
        for k in shell_dict.keys():
            if k not in shell_dict_all.keys():
                shell_dict_all[k]=shell_dict[k]
            else:
                shell_dict_all[k].extend(shell_dict[k])
    with open(out_file,'w') as fp:
        json.dump(shell_dict_all,fp)


def analyze_adfa():
    file_path = "adfa_statis_next_50.json"
    with open(file_path,'r') as fp:
        data = json.load(fp)
    shell=[]
    shell_user=[]
    for key in data.keys():
        if type(data[key])!=dict:
            shell.append(key)
        else:
            shell_user.append(data[key])
    shell_dict = {}
    for sh in shell:
        shell_dict[sh]=cc(data[sh])
    user_squence = []
    for shu in shell_user:  # get user information
        suser = []
        for sk in shu.keys():
            suc = cc(shu[sk])
            s_prb = " "
            for sp in suc.keys():
                cre_prb = (suc[sp]/sum(suc.values()))*(shell_dict[sk][sp]/sum(shell_dict[sk].values()))
                s_prb = s_prb+str(cre_prb)+" "
            suser.append([sk,s_prb])
        user_squence.append(suser)
    np.save("adfa_statis_next_50_prb",user_squence)

def analyze_adfa_double():
    file_path = "adfa_statis_double_50.json"
    with open(file_path,'r') as fp:
        data = json.load(fp)
    shell=[]
    shell_user=[]
    for key in data.keys():
        if type(data[key])!=dict:
            shell.append(key)
        else:
            shell_user.append(data[key])
    shell_dict = {}
    for k in shell:
        shell_dict[k]=cc(["".join(dk) for dk in data[k]])
    user_squence = []
    for shu in shell_user:
        suser = []
        for sk in shu.keys():
            suc = cc(["".join(sku) for sku in shu[sk]])
            s_prb = " "
            for sp in suc.keys():
                cre_prb = (suc[sp]/sum(suc.values()))*(shell_dict[sk][sp]/sum(shell_dict[sk].values()))
                s_prb = s_prb+str(cre_prb)+" "
            suser.append([sk,s_prb])
        user_squence.append(suser)
    np.save("adfa_statis_double_50_prb",user_squence)


def combained_():
    # adfa_data_statis_next_gram
    # adfa_data_statis_next_gram_ratio
    data_next = np.load("adfa_data_type_50_next_gram.npy",allow_pickle=True)
    data = np.load("adfa_data_type_50_vec.npy",allow_pickle=True)
    # next_fea = []
    # for di in data_next:
    #     di_f = []
    #     for dj in np.array(di)[:,1]:
    #         di_f.extend(dj.split())
    #     next_fea.append(np.array(di_f).astype("float64"))
    
    # data_double = np.load("adfa_statis_double_50_prb.npy",allow_pickle=True)
    # double_fea = []
    # for di in data_double:
    #     di_f = []
    #     for dj in np.array(di)[:,1]:
    #         di_f.extend(dj.split())
    #     double_fea.append(np.array(di_f).astype("float64"))
    # dn_fea = [np.hstack((di,dj))[:100] for di,dj in zip(next_fea,double_fea)]
    # pdb.set_trace()
    com_fea = [np.hstack((di,dj)) for di,dj in zip(data,data_next)]
    np.save("adfa_data_type_50_vec_next_gram",com_fea)

def data2vec(model,data):
    print("data2vec....")
    datavec = []
    tmp = []

    # for d in range(len(model.docvecs)):
    #     tmp.append(model.docvecs[d])
    # np.save("adfa_squence",tmp)

    for d in data:
        div = []
        for di in d:
            div.extend(model[di].tolist())
        datavec.append(div)
    return np.array(datavec)



if __name__ == "__main__":
    # file_path = "K:\Guang\code\Training_Data_Master\*"
    # file_path = "K:\Guang\code\Attack_Data_Master\*\*"
    # file2data(file_path)
    # analyze_adfa()
    # analyze_adfa_double()
    combained_()
    # data = np.load("adfa_data_type_50.npy")
    # model = doc2vec_g(data)
    # data_vec = data2vec(model,data)
    # pdb.set_trace()
    # data_next = np.load("adfa_statis_next_50_prb.npy",allow_pickle=True)
    # pdb.set_trace()

