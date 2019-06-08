'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

from dataloader import SubsetOfList
import time
import os

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint')
parser.add_argument('--net', type=str, default="DenseNet121")
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--train-dp', type=float, default=0)
parser.add_argument('--droplayer', type=float, default=0)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--seed', type=int, default=1226)
parser.add_argument('--sidx', type=int, default=0,
                    help='The strating index of samples in each class for training')
parser.add_argument('--eidx', type=int, default=4800,
                    help='The end index of samples in each class for training')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

train_list = torch.load('./datasets/CIFAR10_TRAIN_Split.pth')['clean_train']
trainset = SubsetOfList(train_list, transform=transform_train, start_idx=args.sidx, end_idx=args.eidx)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1000, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = eval(args.net)(train_dp=args.train_dp, droplayer=args.droplayer)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('%s Epoch %d, iteration %d, Loss: %.3f | Acc: %.3f' % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                         epoch, batch_idx, train_loss / (batch_idx + 1),
                                                                         100. * correct / total))


def test(epoch):
    global best_acc
    net.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(cifar_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('* Test on CIFAR10: %s Epoch %d, Loss: %.3f | Acc: %.3f (Notice: droplayer is applied during testing)' % (
            time.strftime("%Y-%m-%d %H:%M:%S"), epoch, test_loss / (batch_idx + 1), 100. * correct / total))

    # Save checkpoint.
    acc = 100. * correct / total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if acc > best_acc:

        if not os.path.isdir('checkpoint-ln'):
            os.mkdir('checkpoint-ln')
        torch.save(state, './checkpoint-ln/cifar10-ckpt-%s-%dto%d-dp%.3f-droplayer%.3f-seed%d.t7' % (args.net,
                                                                                                     args.eidx,
                                                                                                     args.sidx,
                                                                                                     args.train_dp,
                                                                                                     args.droplayer,
                                                                                                     args.seed))
        best_acc = acc

    torch.save(state, './checkpoint-ln/cifar10-ckpt-%s-%dto%d-dp%.3f-droplayer%.3f-seed%d-latest.t7' % (args.net,
                                                                                                        args.eidx,
                                                                                                        args.sidx,
                                                                                                        args.train_dp,
                                                                                                        args.droplayer,
                                                                                                        args.seed))


if args.resume != '':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    test(0)
    exit()

for epoch in range(start_epoch, start_epoch + 300):
    if epoch in [150, 225]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    train(epoch)
    test(epoch)
