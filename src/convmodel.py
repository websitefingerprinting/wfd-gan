import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchsummaryX import summary


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

class Generator(nn.Module):
    def __init__(self, seq_size, latent_dim):
        super(Generator, self).__init__()
        self.seq_size = seq_size
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.fc = nn.Sequential(
            *block(self.latent_dim, 2048, normalize=False),
        )
        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Upsample(scale_factor=4),
            MyConv1dPadSame(256, 256, 4, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MyConv1dPadSame(256, 128, 4, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=4),
            MyConv1dPadSame(128, 128, 4, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MyConv1dPadSame(128, 64, 4, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=4),
            MyConv1dPadSame(64, 64, 4, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            MyConv1dPadSame(64, 1, 4, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.shape[0], 256, 8)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, length):
        super(Discriminator, self).__init__()
        self.length = length
        self.layer1 = nn.Sequential(
            MyConv1dPadSame(1, 32, 4, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            MyConv1dPadSame(32, 32, 4, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            MyMaxPool1dPadSame(4, 1),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            MyConv1dPadSame(32, 64, 4, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            MyConv1dPadSame(64, 64, 4, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            MyMaxPool1dPadSame(4, 1),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            MyConv1dPadSame(64, 128, 4, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            MyConv1dPadSame(128, 128, 4, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            MyConv1dPadSame(128, 256, 4, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            MyMaxPool1dPadSame(4, 1),
            nn.Dropout(0.1)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * 1, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Discriminator(10000).to(device)
    summary(model, torch.zeros(32, 1, 512))

    model = Generator(512, 50)
    summary(model, torch.zeros(32, 50))