'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, train_dp=0, test_dp=0,
                droplayer=0.0, bdp=0):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )
        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp

        self.droplayer = droplayer

    def forward(self, x):
        # disabling droplayer
        action = 0#np.random.binomial(1, self.droplayer)

        if action == 1:
            x = self.shortcut(x)
            odim = self.dense_depth #+ self.out_planes - d
            out = torch.cat([x, torch.zeros(x.size(0), odim, x.size(2), x.size(3)).to(x.device)], 1)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))

            if self.test_dp > 0 or (self.train_dp > 0 and self.training):
                dp = max(self.train_dp, self.test_dp)
                out = F.dropout(out, dp, training=True)

            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = torch.bernoulli(
                    self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            x = self.shortcut(x)
            d = self.out_planes
            out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg, train_dp=0, test_dp=0, droplayer=0, bdp=0, middle_feat_num=1):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64

        # about drop layer
        total_blocks = sum(num_blocks)
        dl_step = droplayer / total_blocks
        self.test_dp = test_dp

        dl_start = 0
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1,
                                    train_dp=train_dp, test_dp=test_dp, dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[0] * dl_step
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2,
                                    train_dp=train_dp, test_dp=test_dp, dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[1] * dl_step
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2,
                                    train_dp=train_dp, test_dp=test_dp, dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += num_blocks[2] * dl_step
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2,
                                    train_dp=train_dp, test_dp=test_dp, dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)
        self.middle_feat_num = middle_feat_num

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride, train_dp=0, test_dp=0,
                    dl_start=0, dl_step=0, bdp=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0,
                                     train_dp=train_dp, test_dp=test_dp, droplayer=dl_start + dl_step*i, bdp=bdp))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

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

def DPN26(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)

def DPN92(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)


def test():
    net = DPN92()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
