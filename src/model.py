import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchsummaryX import summary

'''Conditional WGAN with gradient penalty'''


class Generator(nn.Module):
    def __init__(self, seq_size, class_dim, latent_dim, scaler_min, scaler_max, is_gpu=False):
        super(Generator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.latent_dim = latent_dim
        self.LongTensor = torch.cuda.LongTensor if is_gpu else torch.LongTensor
        self.scaler_min = scaler_min
        self.scaler_max = scaler_max

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.class_dim, 512, normalize=False),
            # *block(512, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, self.seq_size),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        trace = self.model(input)

        # 1) mask the tail of each trace according to the first element which is the learned burst seq length
        # https://discuss.pytorch.org/t/set-value-of-torch-tensor-up-to-some-index/102097
        burst_length = trace[:, 0] * (self.scaler_max - self.scaler_min) + self.scaler_min
        mask = torch.zeros_like(trace)
        mask[(torch.arange(trace.shape[0]), burst_length.type(self.LongTensor) + 1)] = 1
        mask = 1 - mask.cumsum(dim=-1)

        trace = trace * mask
        return trace


class Discriminator(nn.Module):
    def __init__(self, seq_size, class_dim):
        super(Discriminator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.model = nn.Sequential(
            nn.Linear(self.seq_size + self.class_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, trace, c):
        input = torch.cat([trace, c], 1)
        validity = self.model(input)
        return validity


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size, stride_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class DF(nn.Module):
    def __init__(self, length, num_classes=100):
        super(DF, self).__init__()
        self.length = length
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            # nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(1, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Conv1d(64, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            # nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(128, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            MyConv1dPadSame(128, 256, 8, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * self.linear_input(), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc(out)
        return out

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 8))
        return res


if __name__ == '__main__':
    generator = Generator(1400, 100, 500, 0, 100, is_gpu=False)
    summary(generator, torch.zeros((32, 500)), c=torch.zeros(32, 100))