from sklearn.datasets import fetch_openml
import torch
from torch import nn, functional
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD, lr_scheduler
import numpy as np
from custom_layers import LinearCustom, Conv2DCustom
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
from os import makedirs
from os.path import join
import wandb

parser = argparse.ArgumentParser(
    description='Training with MLP MNIST')

parser.add_argument('-batch_size', type=int, default=200)
parser.add_argument('-lr', type=float, default=0.07)
parser.add_argument('-lr_decay', type=float, default=1)  # 1 means lr won't decay
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-hid_size', type=int, default=20)
parser.add_argument('--normal', action='store_true', default=False)
parser.add_argument('--dummy', action='store_true', default=False)
parser.add_argument('--in_grad', action='store_true', default=False)
args = parser.parse_args()

class MLP(nn.Module):
    def __init__(self, input_features, output_features, hidden_size, normal=False):
        super().__init__()
        self.normal = normal
        self.fc1 = nn.Linear(input_features, hidden_size)
        if normal:
            self.fc2 = nn.Linear(hidden_size, output_features)
        else:
            self.fc2 = LinearCustom(hidden_size, output_features)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class Conv(nn.Module):
    def __init__(self, output_classes, normal=False):
        super().__init__()
        self.normal = normal
        k = 5
        out_ch = 8
        s = 2
        if normal:
            self.conv1 = torch.nn.Conv2d(1, out_ch, k, stride=s, padding=0, dilation=1, groups=1, bias=True)
        else:
            self.conv1 = Conv2DCustom(1, out_ch, k, stride=s, padding=0, dilation=1, groups=1, bias=True)

        input_features = (((28 - k) // s) + 1)**2 * out_ch
        self.fc1 = nn.Linear(input_features, output_classes)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.fc1(x.view(x.shape[0], -1))
        return x


if not args.dummy:
    mnist = fetch_openml('mnist_784', version=1, cache=True, data_home='./')
else:
    class Object(object):
        pass
    mnist = Object()
    mnist.data = np.random.randint(0, 255, (70000, 784))
    mnist.target = np.random.randint(0, 10, (70000))


x = torch.Tensor(mnist.data) / 255.0
x = x.view(-1, 1, 28, 28)
y = torch.Tensor(mnist.target.astype(np.int)).long()
x_train, x_val, x_test = x[:50000], x[50000:60000], x[60000:]
y_train, y_val, y_test = y[:50000], y[50000:60000], y[60000:]

train_ds = TensorDataset(x_train, y_train)
if args.in_grad:
    train_ds.tensors[0].requires_grad = True  # for testing
# val_ds = TensorDataset(x_val, y_val)

# model = MLP(784, 10, args.hid_size, args.normal)
model = Conv(10, normal=args.normal)
optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.0001)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_decay ** epoch)

criterion = torch.nn.CrossEntropyLoss()
time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
pbar = tqdm(range(args.epochs))
wandb.init(project='Energy Modulated Dropout', group='mnist')
wandb.config.update(args)
wandb.watch(models=model, log_freq=1)
for epoch in pbar:
    # Doing evaluation first to have performance of untrained model as reference
    model.eval()
    val_preds = torch.max(model(x_val), dim=1)[1].detach().cpu().numpy()
    val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)
    print(val_acc)
    wandb.log({'val accuracy': val_acc}, step=epoch)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size)
    model.train()
    for i, (input, target) in enumerate(train_dl):
        optimizer.zero_grad()
        pred = model(input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        pbar.set_description('loss: {:.4f}'.format(float(loss.data)))

    wandb.log({'train loss': loss}, step=epoch)
    scheduler.step()


