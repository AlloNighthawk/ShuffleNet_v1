import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict


def conv1x1(in_chans, out_chans, n_groups=1):
    return nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, groups=n_groups)

def conv3x3(in_chans, out_chans, stride, n_groups=1):
    # Attention: no matter what the stride is, the padding will always be 1.
    return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=stride, groups=n_groups)

def channel_shuffle(inp, n_groups):
    batch_size, chans, height, width = inp.data.size()
    chans_group = chans // n_groups
    # reshape
    inp = inp.view(batch_size, n_groups, chans_group, height, width)
    inp = torch.transpose(inp, 1, 2).contiguous()
    inp = inp.view(batch_size, -1, height, width)
    return inp


class ShuffleUnit(nn.Module):
    def __init__(self, in_chans, out_chans, stride, n_groups=1):
        super(ShuffleUnit, self).__init__()

        self.bottle_chans = out_chans // 4
        self.n_groups = n_groups

        if stride == 1:
            self.end_op = 'Add'
            self.out_chans = out_chans
        elif stride == 2:
            self.end_op = 'Concat'
            self.out_chans = out_chans - in_chans

        self.unit_1 = nn.Sequential(conv1x1(in_chans, self.bottle_chans, n_groups=n_groups),
                                  nn.BatchNorm2d(self.bottle_chans),
                                  nn.ReLU())
        self.unit_2 = nn.Sequential(conv3x3(self.bottle_chans, self.bottle_chans, stride, n_groups=n_groups),
                                    nn.BatchNorm2d(self.bottle_chans),
                                    conv1x1(self.bottle_chans, self.out_chans, n_groups=n_groups))

    def forward(self, inp):
        if self.end_op == 'Add':
            residual = inp
        else:
            residual = F.avg_pool2d(inp, kernel_size=3, stride=2, padding=1)

        x = self.unit_1(inp)
        x = channel_shuffle(x, self.n_groups)
        x = self.unit_2(x)

        if self.end_op == 'Add':
            return residual + x
        else:
            return torch.cat((residual, x), 1)

class ShuffleNet(nn.Module):
    def __init__(self, n_groups, n_classes):
        super(ShuffleNet, self).__init__()
        self.in_chans = 3
        self.n_groups = n_groups
        self.n_classes = n_classes

        self.conv1 = conv3x3(self.in_chans, 24, 2)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        stage_out_chans_list = [[144, 288, 576], [200, 400, 800], [240, 480, 960],
                                [272, 544, 1088], [384, 768, 1536]]# g=1, 2, 3, 4, 8
        if n_groups > 4:
            stage_out_chans = stage_out_chans_list[-1]
        else:
            stage_out_chans = stage_out_chans_list[n_groups-1]

        # Stage 2
        op = OrderedDict()
        unit_prefix = 'stage_2_unit_'
        op[unit_prefix+'0'] = ShuffleUnit(24, stage_out_chans[0], 2, self.n_groups)
        for i in range(3):
            op[unit_prefix+str(i+1)] = ShuffleUnit(stage_out_chans[0], stage_out_chans[0], 1, self.n_groups)
        self.stage2 = nn.Sequential(op)

        op = OrderedDict()
        unit_prefix = 'stage_3_unit_'
        op[unit_prefix+'0'] = ShuffleUnit(stage_out_chans[0], stage_out_chans[1], 2, self.n_groups)
        for i in range(7):
            op[unit_prefix+str(i+1)] = ShuffleUnit(stage_out_chans[1], stage_out_chans[1], 1, self.n_groups)
        self.stage3 = nn.Sequential(op)

        op = OrderedDict()
        unit_prefix = 'stage_4_unit_'
        op[unit_prefix+'0'] = ShuffleUnit(stage_out_chans[1], stage_out_chans[2], 2, self.n_groups)
        for i in range(3):
            op[unit_prefix+str(i+1)] = ShuffleUnit(stage_out_chans[2], stage_out_chans[2], 1, self.n_groups)
        self.stage4 = nn.Sequential(op)

        self.global_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(stage_out_chans[-1], self.n_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.constant(m.bias, 0)

    def forward(self, inp):
        inp = self.conv1(inp)
        inp = self.maxpool(inp)
        inp = self.stage2(inp)
        inp = self.stage3(inp)
        inp = self.stage4(inp)
        inp = self.global_pool(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.fc(inp)

        return F.log_softmax(inp, dim=1)