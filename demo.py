import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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

def load_adfa_webshell_files(rootdir):
    x = []
    y = []
    allfile=dirlist(rootdir,[])
    for file in allfile:
        # if re.match(r"K:\\Guang\\code\\Attack_Data_Master/Web_Shell_\d+\\UAD-W*", file):
        if "Web_Shell" in file:
            x.append(load_one_flle(file))
            y.append(1)
    return x, y



if __name__ == '__main__':
    x2, y2 = load_adfa_webshell_files(r"K:\Guang\code\Attack_Data_Master")    # 训练集（attack）
    x1, y1 = load_adfa_training_files(r"K:\Guang\code\Training_Data_Master")
     # 训练集（normal）
    
    x3, y3 = load_adfa_training_files(r"K:\Guang\code\Validation_Data_Master")  # 验证集（normal）

    # 训练集黑白样本混合
    x_train = x1 + x2
    y_train = y1 + y2
    x_validate = x3 + x2
    y_validate = y3 + y2

    # n-gram模型
    vectorizer = CountVectorizer(min_df=1, analyzer='word', ngram_range=(2, 3))
    vecmodel = vectorizer.fit(x_train)
    x_train = vecmodel.transform(x_train).toarray()
    x_validate = vecmodel.transform(x_validate).toarray()

    data = np.load("adfa_statis_next_50_prb.npy")
    labels = np.load("adfa_labels_50.npy")
    next_fea = []
    for di in data:
        di_f = []
        for dj in np.array(di)[:,1]:
            di_f.extend(dj.split())
        next_fea.append(np.array(di_f).astype("float64"))

    # 根据训练集生成KNN模型
    clf = KNeighborsClassifier(n_neighbors=4).fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, n_jobs=-1, cv=10)
    # 反映KNN模型训练拟合的程度
    print ("Training accurate: ")
    print (scores)
    print (np.mean(scores))

    # Make predictions using the validate set
    y_pre = clf.predict(x_validate)
    print ("Predict result: " % y_pre)
    # 预测的准确度
    print ("Prediction accurate: %2f" % np.mean(y_pre == y_validate))