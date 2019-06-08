'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, train_dp=0, test_dp=0, bdp=0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        if self.test_dp > 0 or (self.train_dp > 0 and self.training):
            dp = max(self.train_dp, self.test_dp)
            out = F.dropout(out, dp, training=True)
        if self.bdp > 0:
            # each sample will be applied the same mask
            bdp_mask = torch.bernoulli(self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
            out = bdp_mask * out

        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, train_dp=0, test_dp=0, bdp=0):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.test_dp = test_dp

    def _make_dense_layers(self, block, in_planes, nblock, train_dp=0, test_dp=0, bdp=0):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, train_dp=train_dp, test_dp=test_dp, bdp=bdp))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.dense1:
            layer.test_dp = dp
        for layer in self.dense2:
            layer.test_dp = dp
        for layer in self.dense3:
            layer.test_dp = dp
        for layer in self.dense4:
            layer.test_dp = dp

    def penultimate(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)

        return out

    def forward(self, x, penu=False):
        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

def DenseNet105(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return DenseNet(Bottleneck, [6,12,16,16], growth_rate=32, train_dp=train_dp, test_dp=test_dp, bdp=bdp)

def DenseNet121(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, train_dp=train_dp, test_dp=test_dp, bdp=bdp)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
