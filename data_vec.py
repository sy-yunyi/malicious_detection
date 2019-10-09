import os
def dataExtract(file_path):
    header = ["User-Agent:","Pragma:","Cache-control:","Accept:","Accept-Encoding:","Accept-Charset:","Accept-Language:","Host:","Cookie:","Connection:","Content-Length:","Content-Type:"]
    data_list = []
    with open(file_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line != "\n" and line.split()[0] not in header and not line.startswith("Accept"):
                data_list.append(line.strip())

    with open("data/data_anomalousTrafficTest","w") as pf:
        pf.write(data_list)
    # data_g = (d for d in data_list)
    # data_e = []
    # for d in data_g:
    #     if d.split()[0] == "GET":
    #         data_e.append(d)
    #     elif d.split()[0] == "POST" or d.split()[0] == "PUT":
    #         data_e.append([d,data_g.__next__()])
    
    # for i in data_e:
    #     print(i)
        
                    



if __name__ == "__main__":
    file_path = r"F:\广电\source\detection_web_attack\data\anomalousTrafficTest.txt"
    dataExtract(file_path)


