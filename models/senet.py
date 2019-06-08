'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp
        self.droplayer = droplayer

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        action = np.random.binomial(1, self.droplayer)
        if action == 1:
            out = shortcut
        else:

            out = self.conv1(out)

            if self.test_dp > 0 or (self.train_dp > 0 and self.training):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)
            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = torch.bernoulli(
                    self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
                out = bdp_mask * out

            out = self.conv2(F.relu(self.bn2(out)))

            # Squeeze
            w = F.avg_pool2d(out, out.size(2))
            w = F.relu(self.fc1(w))
            w = F.sigmoid(self.fc2(w))
            # Excitation
            out = out * w

            out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0, middle_feat_num=1):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        nblks = sum(num_blocks)
        dl_step = droplayer / nblks

        dl_start = 0
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[0] * dl_step
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[1] * dl_step
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[2] * dl_step
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)
        self.linear = nn.Linear(512, num_classes)

        self.test_dp = test_dp
        self.middle_feat_num = middle_feat_num

    def get_block_feats(self, x):
        feat_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # feat_list.append(out)

        out = self.layer2(out)
        # feat_list.append(out)

        out = self.layer3(out)
        # feat_list.append(out)

        # out = self.layer4(out)
        for nl, layer in enumerate(self.layer4):
            out = layer(out)
            if len(self.layer4) - nl - 1 <= self.middle_feat_num and len(self.layer4) - nl - 1 > 0:
                feat_list.append(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

    def _make_layer(self, block, planes, num_blocks, stride, train_dp=0, test_dp=0, dl_start=0, dl_step=0, bdp=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for si, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, train_dp=train_dp, test_dp=test_dp,
                                droplayer=dl_start+dl_step*si, bdp=bdp))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.layer1:
            layer.test_dp = dp
        for layer in self.layer2:
            layer.test_dp = dp
        for layer in self.layer3:
            layer.test_dp = dp
        for layer in self.layer4:
            layer.test_dp = dp

    def forward(self, x, penu=False, block=False):
        if block:
            return self.get_block_feats(x)

        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

def SENet18(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return SENet(PreActBlock, [2,2,2,2], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)


def test():
    net = SENet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
