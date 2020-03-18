'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
from vgg import VGG
from custom_layers import conf
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='Momentum')
parser.add_argument('--optim', default='adam', choices=['sgd', 'adam'], help='momentum')
parser.add_argument('--batch', default=128, type=int, help='Batch size')
parser.add_argument('--epochs', default=100, type=int, help='Epochs to train for')
parser.add_argument('--workers', default=4, type=int, help='Dataloader workers')
parser.add_argument('--normal', action='store_true', default=False, help='Use pytorch\'s conv layer')
parser.add_argument('--topk', action='store_true', default=False, help='whether to mask topk gradients')
parser.add_argument('--topk_ratio', default=0.25, help='Ratio to be masked')
parser.add_argument('--model', choices=['VGG_tiny', 'VGG_mini', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'sp1'], default='VGG_tiny', help='Pick a VGG')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Device: ', device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
print('Timestamp: ', timestamp)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
conf['topk'] = args.topk
conf['topk_ratio'] = args.topk_ratio
print('==> Building model..')
net = VGG(args.model, normal=args.normal)
print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
net = net.to(device)

wandb.init(name=timestamp, project='Energy Modulated Dropout', group='cifar')
wandb.config.update(args)
criterion = nn.CrossEntropyLoss()
wandb.watch(models=net, criterion=criterion, log='all')

if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.decay)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # + shapes_kernel_loss(net)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({'Train Loss': train_loss/(batch_idx+1)}, step=epoch)
    wandb.log({'Train Acc.': 100.*correct/total}, step=epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        wandb.log({'Test Loss': test_loss/(batch_idx+1)}, step=epoch)
        wandb.log({'Test Acc.': 100.*correct/total}, step=epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'timestamp': timestamp,
            **vars(args)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + timestamp + '.pth')
        best_acc = acc


for epoch in tqdm(range(start_epoch, start_epoch+args.epochs)):
    train(epoch)
    test(epoch)

wandb.log({'hparam/accuracy': best_acc})
