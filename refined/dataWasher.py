import pandas as pd
import torch
import numpy as np

slectedNot = [
    "Q7-set_no",
    "Q8-game_no",
    "Q9-point_no",
    "Q16-server",
    "Q17-serve_no",
    "Q18-point_victor",
]
oridd = pd.read_excel("../problems/newbe/1-19/7.xlsx")
columns=oridd.columns
orid = pd.read_excel("../problems/newbe/31/31.xlsx")
dtst = pd.read_excel("../problems/newbe/dtst/mat7.xlsx")
slected = dtst.columns.values
comp = np.zeros((0, 0))
print(type(dtst), dtst)
print(dtst.values[:-1, :])
print(type(dtst), orid)
print(orid.values[:, :])
print(columns.values)
i = 0
j = 0
complist = []
lllist = []
while i < len(columns.values):
    if columns.values[i] == slected[j]:
        complist.append(np.concatenate([orid.values[:, i : i + 1],np.array([[0]])],axis=0))
        lllist.append(slected[j])
        j += 1
        if j >= len(slected):
            break
    i += 1
i = 0
j = 15
while i < len(columns.values):
    if columns.values[i] == slected[j] and not (slected[j] in lllist):
        complist.append(np.concatenate([orid.values[:, i : i + 1],np.array([[0]])],axis=0))
        j += 1
        if j >= len(slected):
            break
    i += 1
j = 0
while j < len(slected):
    if slected[j] in slectedNot:
        j += 1
        continue
    print(slected[j])
    complist[j][1:,:]=complist[j][:-1,:]
    complist[j][0,:]=np.array([0])
    j += 1
print(i, j)
comp = orid.values[:, 0:1]
# comp=np.concatenate((orid.values[:,1:2],comp),axis=1)
comp = np.concatenate(complist, axis=1)
print(len(complist), len(slected))
print(slected)
print(lllist)
print(dtst.values, comp, sep="\n")
print(dtst.values.shape, comp.shape)


rpt=pd.DataFrame(comp)
rpt.columns=slected
print(rpt)
rpt.to_excel("../problems/newbe/dtst/mat31.xlsx",index=False)
