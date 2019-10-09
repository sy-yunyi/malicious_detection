import os
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
        data_e = []
        for d in data_g:
            if d.split()[0] == "GET":
                data_e.append(d)
            elif d.split()[0] == "POST" or d.split()[0] == "PUT":
                data_e.append([d,data_g.__next__()])
    
    for i in data_e:
        print(i)
        
                    



if __name__ == "__main__":
    # 将目标数据才从源数据中提取出来
    file_path = r"F:\广电\source\detection_web_attack\data\normalTrafficTest.txt"
    out_path = r"malicious_detection\data\data_normalTrafficTest"
    dataExtract(file_path,out_path)


