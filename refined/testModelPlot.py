import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from parameter import CHECKPOINT_PATH, IS_CREATE, EPOCH_SIZE


# class Net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.lstm = nn.LSTM(11, 128)
#         self.Linear = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("LinearLayer", nn.Linear(128, 2)),
#                     ("activation", nn.Softplus()),
#                 ]
#             )
#         )


#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.Linear(x)
# return x
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(15, 128)
        self.Linear = nn.Sequential(
            OrderedDict(
                [
                    ("LinearLayer0", nn.Linear(128, 64)),
                    ("activation0", nn.LogSoftmax()),
                    ("LinearLayer1", nn.Linear(64, 32)),
                    ("activation1", nn.LogSoftmax()),
                    ("LinearLayer2", nn.Linear(32, 2)),
                    ("activation2", nn.LogSoftmax()),
                ]
            )
        )
        self.convert = nn.Sequential(
            OrderedDict(
                [
                    ("LinearLayer3", nn.Linear(128, 15)),
                    ("activation4", nn.LogSoftmax()),
                ]
            )
        )

    def forward(self, x):
        x, tup = self.lstm(x)
        x, _ = self.lstm(self.convert(x), tup)
        x = self.Linear(x)
        return x


model = Net().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
with open(CHECKPOINT_PATH + "epoch.log", "r") as file:
    startEpoch = int(file.read())
    print(startEpoch)
checkpoint = torch.load(CHECKPOINT_PATH + f"cnnCheckPoint{startEpoch}")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
model.eval()


data = torch.load("mat31.tsdt")

inputData = data[:, :15].to("cuda")
outputData = (
    torch.tensor((data[:, 15:] - 1).tolist(), dtype=torch.int64).view(-1).to("cuda")
)
# print(ans[:5])
# print(outputData[:5])

# test
outputPredict = model(inputData)
loss = criterion(outputPredict, outputData)
print(loss.item())
cnt = 0
X1 = []
X0 = []
for i in range(len(inputData)):
    # print(i, outputPredict[i], outputData[i])
    # print(criterion(outputPredict[i], outputData[i]))
    lst = outputPredict[i].tolist()
    explt = torch.exp(torch.tensor(lst))
    print(lst.index(max(lst)), outputData[i], torch.exp(torch.tensor(lst)))
    X0.append(explt[0])
    X1.append(explt[1])
    if lst.index(max(lst)) == outputData[i]:
        cnt += 1
print(cnt / len(inputData))

import matplotlib.pyplot as plt

X0 = [itm - 0.5 for itm in X0]
X1 = [itm - 0.5 for itm in X1]
plt.scatter([i for i in range(len(X0))], X0)
plt.scatter([i for i in range(len(X0))], X1)
plt.show()
Y0 = [
    (X0[i] + X0[i + 1] + X0[i - 1] + X0[i - 2] + X0[i + 2] + X0[i - 3] + X0[i + 3]) / 7
    for i in range(3, len(X0) - 3)
]
Y0_3 = [(X0[i] + X0[i + 1] + X0[i - 1]) / 3 for i in range(1, len(X0) - 1)]
Y0_5 = [
    (X0[i] + X0[i + 1] + X0[i - 1] + X0[i - 2] + X0[i + 2]) / 5
    for i in range(2, len(X0) - 2)
]
Y0_9 = [
    (
        X0[i]
        + X0[i + 1]
        + X0[i - 1]
        + X0[i - 2]
        + X0[i + 2]
        + X0[i - 3]
        + X0[i + 3]
        + X0[i - 4]
        + X0[i + 4]
    )
    / 9
    for i in range(4, len(X0) - 4)
]
# Y1=[i for i in range(1,len(X1)-1)]
# Y0[0]=X0[0]
# Y1[0]=X1[0]
# for i in range(1,len(X0)):
#     Y0[i]=Y0[i-1]+X0[i]
#     Y1[i]=Y1[i-1]+X1[i]
plt.scatter([i for i in range(len(Y0))], Y0)
# plt.scatter([i for i in range(len(Y0))],Y1)
plt.show()

plt.plot(Y0)
# plt.scatter([i for i in range(len(Y0))],Y1)
plt.show()
plt.plot(Y0_3)
# plt.scatter([i for i in range(len(Y0))],Y1)
plt.show()
plt.plot(Y0_5)
# plt.scatter([i for i in range(len(Y0))],Y1)
plt.show()
plt.plot([i-7 for i in range(len(Y0_9))],Y0_9)
# plt.scatter([i for i in range(len(Y0))],Y1)
import numpy as np

plt.plot([i-7 for i in range(len(Y0_9))],(sum(Y0_9) / len(Y0_9)) * np.ones(len(Y0_9)))
# plt.scatter([i for i in range(len(Y0))],Y1)
plt.show()
