

##############Need to pre-divide the independent test set
##############The data here is the file that has been divided into the training set(id_prop.csv)




import pandas as pd
import numpy as np
def sbatch():  
    """
    """
    import os, sys,time
    ela = []
    sblist = []
    for root,dirs,files in os.walk(r"./"):
        for dir in dirs:
            ela.append(os.path.join(root,dir))
    mine = os.getcwd()
    cmd = mine +"/gpu_tesk.sh"   
    for i in ela:    
        os.chdir(mine)
        #print(i)
        os.chdir(i)
        p = os.popen('pwd')
        print(p.read())
        p.close()
        try:
            os.system("sbatch "+ cmd)
            print("sbatch it ",i)
            time.sleep(60)
        except:
            sblist.append(i)
        os.chdir(mine)
    os.chdir(mine)
    print(sblist)
    return sblist
def native_cp(str_log_open,str_new):
    import os
    import shutil
    str_log_open = str_log_open 
    shutil.copyfile(str_log_open, str_new)

def change(a,b):
    with open('gpu_tesk.sh','r',encoding='utf-8') as f:  
        lines=[] # 
        for line in f.readlines():
            if line!='\n':
                lines.append(line)
    f.close()
    with open('gpu_tesk.sh','w',encoding='utf-8') as f:
        for line in lines:
            if a in line:
                line = b 
                f.write('%s\n' %line)
            else:
                f.write('%s' %line)
sblist = []
for i in ["a","b","c","d"]:
    for j in [100,1000,10000]:
    ########################train dataset#########################
        df = pd.read_csv("./datasettest/task"+str(i)+str(j)+".csv")
        did = df.iloc[:,0].values
        dt = df.iloc[:,-1].values
        did = pd.DataFrame(did)
        dt  = pd.DataFrame(dt) 
        did["dt"] =dt
        did.to_csv("./tmp.csv",index=None)
        look = pd.read_csv("./tmp.csv",header = None)
        look = look.iloc[1:,:]
        look.to_csv("./id_prop.csv",index=None,header = None)
        native_cp("./id_prop.csv","./all_cif/id_prop.csv")      ######id prop
        s = j*0.6  
        t = j*0.2
        h = j*0.2
        change("#SBATCH -J ","#SBATCH -J "+str(i)+str(j))
        change("python ../main.py","python ../main.py  --task classification --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 ./all_cif > torchlog_cgcnn_"+str(i)+str(j))
        
        ##############vim the bash
        #############################################       
        import os, sys,time
        mine = os.getcwd()
        cmd = mine +"/gpu_tesk.sh"
        os.chdir(mine)
        #print(i)
        #os.chdir(i)  ##
        try:
            os.system("sbatch "+ cmd)
            print("sbatch it ",i,j)
            time.sleep(10)
            #os.system("ls")
            #print("ls")
        except:
            sblist.append(i)
        print(sblist)
        
        
        
        
        