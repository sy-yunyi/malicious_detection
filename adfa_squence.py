import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import groupBy 
import pdb


def load_one_flle(filename):
    x = []
    with open(filename,'r',encoding="windows-1252") as f:
        line = f.readline()
        line = line.strip('\n')
    return line

def load_adfa_training_files(rootdir):
    x = []
    y = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_flle(path))
            y.append(0)
    return x, y

def dirlist(path, allfile):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def load_type_file(filename):
    with open(filename,'r',encoding="windows-1252") as f:
        line = f.readline()
        line = line.strip('\n').split()
    return groupBy(line,50,50),len(groupBy(line,50,50))


def load_adfa_type_files(rootdir,att_type=None):
    x = []
    y = []
    allfile=dirlist(rootdir,[])
    attack_tyep = {"Adduser":1,"Hydra":2,"Java":3,"Meter":4,"Web":5}
    for file in allfile:
        # if re.match(r"K:\\Guang\\code\\Attack_Data_Master/Web_Shell_\d+\\UAD-W*", file):
        for ty in attack_tyep.keys():
            if ty in file:
                data,lb = load_type_file(file)
                x.extend(data)
                y.extend([attack_tyep[ty]] * lb)
        if "Attack" not in file:
            data,lb = load_type_file(file)
            x.extend(data)
            y.extend([0] * lb)
    return x, y

def next_gram():
    data = np.load("adfa_data_type_50.npy")
    labels = np.load("adfa_labels_type.npy")
    data = [" ".join(d) for d in data]

    vectorizer = CountVectorizer(min_df=1, analyzer='word', ngram_range=(2, 2))
    vecmodel = vectorizer.fit(data)
    x_train = vecmodel.transform(data).toarray()

    data_next = x_train / sum(x_train)

    pdb.set_trace()


def tfidf_adfa():
    data = np.load("adfa_data_50.npy")
    labels = np.load("adfa_labels_50.npy")
    data = [" ".join(d) for d in data]

    vectorizer = TfidfVectorizer(min_df=1)
    vecmodel = vectorizer.fit(data) # 按照训练集训练vocab词汇表
    print ("vocabulary_: ")
    print (vecmodel.vocabulary_)

    x_train = vecmodel.transform(data).toarray()

    pdb.set_trace()


if __name__ == '__main__':
    next_gram()
    # tfidf_adfa()
    # x2, y2 = load_adfa_type_files(r"K:\Guang\code\Attack_Data_Master")    # 训练集（attack）
    # x1, y1 = load_adfa_type_files(r"K:\Guang\code\Training_Data_Master")
    #  # 训练集（normal）
    # pdb.set_trace()
    
    # x3, y3 = load_adfa_type_files(r"K:\Guang\code\Validation_Data_Master")  # 验证集（normal）
