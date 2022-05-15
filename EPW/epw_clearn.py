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