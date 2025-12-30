import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

# small network
model = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# load mnist
train = dsets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

# store an image for inference testing
x, y = train[0]
x = x.view(-1)

print("label for inference: ", y)
with open("mnist_inf.bin", "wb") as f:
    f.write(x.numpy().astype(np.float32).tobytes())

# optimizer & loss
opt = torch.optim.SGD(model.parameters(), lr = 0.1)
loss_fn = nn.CrossEntropyLoss()

# train a few epochs
for _ in range(5):
    for x, y, in loader:
        x = x.view(x.size(0), -1)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

state = model.state_dict()
with open("mnist_mlp.bin", "wb") as f:
    for k, v in state.items():
        f.write(v.cpu().numpy().astype(np.float32).tobytes())
