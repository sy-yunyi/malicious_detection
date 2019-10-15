import os
import re
import urllib
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

def dataVec(file_path):
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
            print(data_set)
        
    # for d in data_e:        
    #     if len(d) == 1:
    #         print(d)
    #         d = d[0].split()
    #         print(" ".join([d[0]," ".join(re.split("[?=&]",d[1].split("/")[-1]))]))
    #     elif len(d) == 2:
    #         print(d)
    #         ac_end = re.split("[ /]",d[0]) # 0:post,-3:end
    #         args = urllib.parse.unquote(urllib.parse.unquote(d[1]))
    #         print(" ".join([ac_end[0],ac_end[-3]," ".join(re.split("[?=&]",args))]))

            
            
                

            
        
                    



if __name__ == "__main__":
    # 将目标数据才从源数据中提取出来
    # file_path = r"F:\广电\source\detection_web_attack\data\normalTrafficTest.txt"
    # out_path = r"malicious_detection\data\data_normalTrafficTest"
    # dataExtract(file_path,out_path)

    #数据向量化
    file_path = r"D:\six\code\malicious_detection\data\data_normalTrafficTest"
    dataVec(file_path)


