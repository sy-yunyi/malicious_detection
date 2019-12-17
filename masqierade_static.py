import numpy as np 
import json
import pdb
from collections import Counter as cc

def sequenceStatis(data,trim=False):
    data = np.array(data).reshape(7500,100)
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
        shell_dict_all[j]=shell_dict
        for k in shell_dict.keys():
            if k not in shell_dict_all.keys():
                shell_dict_all[k]=shell_dict[k]
            else:
                shell_dict_all[k].extend(shell_dict[k])
    with open("masq_statis_double.json",'w') as fp:
        json.dump(shell_dict_all,fp)


def analyze_masq():
    file_path = "masq_statis_next.json"
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
    # np.save("masq_statis_next_prb",user_squence)
            

def analyze_masq_double():
    file_path = "masq_statis_double.json"
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
    np.save("masq_statis_double_prb",user_squence)

def combained_(data):
    data_next = np.load("masq_statis_next_prb.npy",allow_pickle=True)
    data = np.array(data).reshape(7500,100)
    data1 = np.load("masq_data.npy",allow_pickle=True)
    next_fea = []
    for di in data_next:
        di_f = []
        for dj in np.array(di)[:,1]:
            di_f.extend(dj.split())
        next_fea.append(np.array(di_f).astype("float32"))
    
    data_double = np.load("masq_statis_double_prb.npy",allow_pickle=True)
    double_fea = []
    for di in data_double:
        di_f = []
        for dj in np.array(di)[:,1]:
            di_f.extend(dj.split())
        double_fea.append(np.array(di_f).astype("float32"))
    dn_fea = [np.hstack((di,dj)) for di,dj in zip(next_fea,double_fea)]
    np.save("masq_data_statis_only",dn_fea)

    # com_fea = [np.hstack((di,dj)) for di,dj in zip(data1,dn_fea)]

    # np.save("masq_data_statis",com_fea)
    pdb.set_trace()



if __name__ == "__main__":
    # analyze_masq_double()
    pass
    
