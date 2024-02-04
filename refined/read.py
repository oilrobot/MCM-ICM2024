import pandas as pd
import torch
import numpy as np

dtst = pd.read_excel("../problems/newbe/dtst/mat31.xlsx")
print(type(dtst), dtst)
print(dtst.values[:-1,:])

# train_tensor = torch.tensor(train.values)
# print(train.values[:, 4:-7].shape)
# print(type(train.columns.values))
data = torch.from_numpy(np.array(dtst.values[:-1,:], dtype=np.float32))
print(data.shape)
torch.save(data, "mat31.tsdt")
