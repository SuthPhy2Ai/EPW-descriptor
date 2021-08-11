#coding=utf-8
from EPW import feature_extraction
import pandas as pd
import numpy as np
df_1 = pd.DataFrame()
# df_temp = pd.DataFrame(np.array(list_a).reshape(1,-1))
# df_1 = df_1.append(df_temp)
# df_1
df_2 = pd.DataFrame()
data_list = pd.read_csv("./your_path_to_cif/cif.csv",engine='python')
task_id = data_list.iloc[:,0]
real_list = []
errorlist = []
q = 0

for i in task_id :
    
    q = q + 1
    
   # if q%100  == 0
    print(q*100/len(task_id),"%")
    
    
    try :
        tem_path = "./your_path_to_cif/cif.csv/" + str(i) + ".cif.vasp"
        get = feature_extraction(tem_path)
        get.read_poscar()
        get.real_direction()
        #print(get.overlap)
        #print(get.overlap_avg)
        #cif_PEW[i] = i
        one_sample_feature =  get.overlap+get.overlap_avg
        df_temp = pd.DataFrame(np.array(one_sample_feature).reshape(1,-1))
        df_1 = df_1.append(df_temp)
        print("succeed",i)
        
    except :
        print("no mpcif",i)
        errorlist.append(i)
        
        

df_1.to_csv("EPW.csv")
print(errorlist)
