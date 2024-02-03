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
        self.lstm = nn.LSTM(11, 128)
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
                    ("LinearLayer3", nn.Linear(128, 11)),
                    ("activation4", nn.LogSoftmax()),
                ]
            )
        )

    def forward(self, x):
        x, tup = self.lstm(x)
        x,_=self.lstm(self.convert(x),tup)
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


data = torch.load("modifiedDataFromColumnEToColumnP.tsdt")

inputData = data[:6000, :11].to("cuda")
outputData = (
    torch.tensor((data[:6000, 11:12] - 1).tolist(), dtype=torch.int64)
    .view(-1)
    .to("cuda")
)
# print(ans[:5])
# print(outputData[:5])

# test
outputPredict = model(inputData)
loss = criterion(outputPredict, outputData)
print(loss.item())
cnt = 0
for i in range(1284):
    # print(i, outputPredict[i], outputData[i])
    # print(criterion(outputPredict[i], outputData[i]))
    lst = outputPredict[i].tolist()
    # print(lst.index(max(lst)), outputData[i])
    if lst.index(max(lst)) == outputData[i]:
        cnt += 1
print(cnt / 1284)
