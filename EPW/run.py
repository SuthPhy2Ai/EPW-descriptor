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


import pandas as pd
import numpy as np
from tqdm import tqdm
df = pd.read_csv("./EPW.csv")
"""
For default 12 high symmetry points conversion
"""
def trans_to_array(df,Row_index):   
    #print("ROW",Row_index)
    for i in range(12):   ## 12 hkpts
        if i == 0:
            deal_with = [float(x) for x in (df["0"][Row_index][1:-1].split())]
        else:
            deal_with+=([float(x) for x in (df[str(i)][Row_index][1:-1].split())])
        #print(i)
    return np.array(deal_with)
#  a zeros numpy array
import itertools
tmp = np.zeros((len(df),12*4))
for i in tqdm(range(len(df))):
    #print(i)
    tmp[i] = trans_to_array(df,i)
clearn_epw = pd.DataFrame(tmp)

for i, j in itertools.product(tqdm(range(len(clearn_epw))[:100]), range(12*4)):
    clearn_epw.iloc[i, j] = -15 if clearn_epw.iloc[i, j] <= 0.000001 else np.log10(clearn_epw.iloc[i, j])

clearn_epw.to_csv("clearn_epw.csv",index=False)
