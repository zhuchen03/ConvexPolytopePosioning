'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, train_dp=0, test_dp=0, bdp=0):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp


    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        ret = torch.cat([y1,y2,y3,y4], 1)
        # if self.test_dp > 0 and (self.train_dp > 0 or self.training):
        #     dp = max(self.train_dp, self.test_dp)
        #     ret = F.dropout(ret, dp, training=True)

        if self.test_dp > 0:
            # in this branch, train_dp means the model's dropout used in the training
            # used to allow using a different dropout at test time, but use the same scaling factor as training
            ret = F.dropout(ret, self.test_dp, training=True) / (1 + self.test_dp) * (1 + self.train_dp)
        elif self.train_dp > 0 and self.training:
            ret = F.dropout(ret, self.train_dp, training=True)

        if self.bdp > 0:
            # each sample will be applied the same mask
            bdp_mask = torch.bernoulli(
                self.bdp * torch.ones(1, ret.size(1), ret.size(2), ret.size(3)).to(ret.device)) / self.bdp
            ret = bdp_mask * ret

        return ret


class GoogLeNet(nn.Module):
    def __init__(self, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(GoogLeNet, self).__init__()

        # not implemented here

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, train_dp=train_dp, test_dp=test_dp, bdp=bdp)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, train_dp=train_dp, test_dp=test_dp, bdp=bdp)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, train_dp=train_dp, test_dp=test_dp, bdp=bdp)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp

    def penultimate(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

    def set_testdp(self, dp):
        self.a3.test_dp = dp
        self.b3.test_dp = dp
        self.a4.test_dp = dp
        self.b4.test_dp = dp
        self.c4.test_dp = dp
        self.d4.test_dp = dp
        self.e4.test_dp = dp
        self.a5.test_dp = dp
        self.b5.test_dp = dp

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


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
