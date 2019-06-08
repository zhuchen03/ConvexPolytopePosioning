'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp
        self.droplayer = droplayer

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if action == 1:
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            if self.test_dp > 0 or (self.train_dp > 0 and self.training):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)
            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = torch.bernoulli(
                    self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, train_dp=0, test_dp=0, droplayer=0,
                                            bdp=0, middle_feat_num=1):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        dl_step = droplayer / sum(num_blocks)

        dl_start = 0
        self.layer1 = self._make_layer(num_blocks[0], 1, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += dl_step * num_blocks[0]
        self.layer2 = self._make_layer(num_blocks[1], 2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += dl_step * num_blocks[1]
        self.layer3 = self._make_layer(num_blocks[2], 2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)
        self.test_dp = test_dp
        self.middle_feat_num = middle_feat_num

    def set_testdp(self, dp):
        for layer in self.layer1:
            layer.test_dp = dp
        for layer in self.layer2:
            layer.test_dp = dp
        for layer in self.layer3:
            layer.test_dp = dp

    def _make_layer(self, num_blocks, stride, train_dp=0, test_dp=0, dl_start=0, dl_step=0, bdp=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for ns, stride in enumerate(strides):
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride,
                                train_dp=train_dp, test_dp=test_dp, droplayer=dl_start+ns*dl_step, bdp=bdp))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        return out

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def forward(self, x, penu=False, block=False):
        if block:
            return self.get_block_feats(x)

        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_block_feats(self, x):
        feat_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # feat_list.append(out)
        out = self.layer2(out)
        # feat_list.append(out)
        for nl, layer in enumerate(self.layer3):
            out = layer(out)
            if len(self.layer3) - nl - 1 <= self.middle_feat_num and len(self.layer3) - nl - 1 > 0:
                feat_list.append(out)

        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]


def ResNeXt29_2x64d(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, train_dp=train_dp, test_dp=test_dp,
                   droplayer=droplayer, bdp=bdp)

def ResNeXt29_4x64d(train_dp=0, test_dp=0, droplayer=0):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, train_dp=train_dp, test_dp=test_dp,
                   droplayer=droplayer)

def ResNeXt29_8x64d(train_dp=0, test_dp=0, droplayer=0):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, train_dp=train_dp, test_dp=test_dp,
                   droplayer=droplayer)

def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)

def test_resnext():
    net = ResNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test_resnext()
