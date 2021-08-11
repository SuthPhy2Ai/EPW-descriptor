from dscribe.descriptors import SOAP
import numpy as np
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.io import read
species = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
rcut = 4.0
nmax = 3
lmax = 3
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


SOAP = pd.DataFrame()
sblist = []
golist = []
a = 0
asum = len(df.iloc[:,0])
for i in df.iloc[:,0]:
    
    a = a+1
    c = read("./all_cif/"+i+".cif")
    try:
        soapc = average_soap.create(c)
        df_temp = pd.DataFrame(np.array(soapc).reshape(1,-1))
        SOAP = SOAP.append(df_temp)
        golist.append(i)
        print(a/asum)
    except:
        sblist.append(i)
        print("sb cif id is",i)
SOAP["mpid"] = golist
print(sblist)


SOAP.to_csv("SOAP.csv",index=None)