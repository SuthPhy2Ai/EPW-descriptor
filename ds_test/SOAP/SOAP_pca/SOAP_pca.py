from dscribe.descriptors import SOAP
import numpy as np
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.io import read
species = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
rcut = 4.0
nmax = 5
lmax = 4
average_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="inner",
    sparse=False
)


import pandas as pd
df = pd.read_csv("./id_prop.csv",header = None)
SOAP_1000 = pd.DataFrame()
faillist = []
golist = []
a = 0
asum = len(df.iloc[:1000,0])
for i in df.iloc[:1000,0]:
    print(i)
    a = a+1
    c = read("../all_cif/"+i+".cif")
    try:
        soapc = average_soap.create(c)
        print(soapc)
        df_temp = pd.DataFrame(np.array(soapc).reshape(1,-1))
        SOAP_1000 = SOAP_1000.append(df_temp)
        golist.append(i)
        print(a/asum)
    except:
        faillist.append(i)
        print("fail cif id is",i)
SOAP_1000["mpid"] = golist
print(faillist)
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(SOAP_1000.iloc[:,:-1])
principalDf = pd.DataFrame(data = principalComponents)

SOAP_else = pd.DataFrame()
faillist = []
golist = []
a = 0
asum = len(df.iloc[:,0])
for i in df.iloc[:,0]:
    print(i)
    a = a+1
    c = read("../all_cif/"+i+".cif")
    try:
        #soapc = average_soap.create(c)
        #print(soapc)
        tmp = pd.DataFrame(np.array(average_soap.create(c).reshape(1,-1)))
        trs = pca.transform(tmp)
        df_temp = pd.DataFrame(np.array(trs).reshape(1,-1))
        SOAP_else = SOAP_else.append(df_temp)
        golist.append(i)
        print(a/asum)
    except:
        faillist.append(i)
        print("fail cif id is",i)
SOAP_else["mpid"] = golist
print(faillist)
#principalDf
SOAP_else.to_csv("SOAP_pca.csv",index=None)