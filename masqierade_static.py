import numpy as np 
import json


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


