import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from parameter import CHECKPOINT_PATH, IS_CREATE, EPOCH_SIZE


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


if IS_CREATE:
    net = Net().to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": None,
        },
        CHECKPOINT_PATH + f"cnnCheckPoint{0}",
    )
    with open(CHECKPOINT_PATH + "epoch.log", "w") as file:
        file.write(f"{0}")

model = Net().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
with open(CHECKPOINT_PATH + "epoch.log", "r") as file:
    startEpoch = int(file.read())
    print(startEpoch)
checkpoint = torch.load(CHECKPOINT_PATH + f"cnnCheckPoint{startEpoch}")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
model.train()

inputData = []
outputData = []
for lmd in range(1, 20):
    data = torch.load(f"mat{lmd}.tsdt")

    inputData.append(data[:, :15].to("cuda"))
    outputData.append(
        (
            torch.tensor((data[:, 15:] - 1).tolist(), dtype=torch.int64)
            .view(-1)
            .to("cuda")
        )
    )

print(len(inputData))

# print(ans[:5])
# print(outputData[:5])

# train
for i in range(EPOCH_SIZE):
    optimizer.zero_grad()
    for lmd in range(0, 5):
        outputPredict = model(inputData[lmd])
        loss = criterion(outputPredict, outputData[lmd])
        loss.backward()
    optimizer.step()
    print(i, loss.item())

torch.save(
    {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": None,
    },
    CHECKPOINT_PATH + f"cnnCheckPoint{epoch+1}",
)
with open(CHECKPOINT_PATH + "epoch.log", "w") as file:
    file.write(f"{epoch+1}")
