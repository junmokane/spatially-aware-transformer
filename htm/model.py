import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class PositionwiseFF(nn.Module):
    def __init__(self, args):
        super(PositionwiseFF, self).__init__()
        dim = args.htm.dim
        hidden_dim = args.htm.mlp.hidden_dim
        dropout = args.htm.mlp.dropout

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, stride=9)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32*3*3 , args.hcam.dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        # x = F.relu(self.conv3(x))
        # print(x.size())
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        return x

class CNNXL(nn.Module):
    def __init__(self, args):
        super(CNNXL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256, args.hcam.dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        return x


class DeCNNXL(nn.Module):
    def __init__(self, args):
        super(DeCNNXL, self).__init__()
        self._net = nn.Sequential(
                nn.Linear(args.hcam.dim, 256),
                nn.ReLU(),
                Unflatten(-1, 64, 2, 2),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2), 
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2), 
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=8, stride=4), 
                nn.Tanh(),
                Deprocess_img())
        
    def forward(self, x):
        x = self._net(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class CNNLarge(nn.Module):
    def __init__(self, args):
        super(CNNLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, stride=9)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32*3*3 , args.hcam.dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        return x


class DeCNNLarge(nn.Module):
    def __init__(self, args):
        super(DeCNNLarge, self).__init__()
        self._net = nn.Sequential(
                nn.Linear(args.hcam.dim, 128),
                nn.ReLU(),
                Unflatten(-1, 32, 2, 2),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1), # 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=9, stride=9), # 11x11
                nn.Tanh(),
                Deprocess_img())
        
    def forward(self, x):
        x = self._net(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=3, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


class Deprocess_img(nn.Module):
    def forward(self, x):
        return (x + 1) / 2


class DeCNN(nn.Module):
    def __init__(self, args):
        super(DeCNN, self).__init__()
        self._net = nn.Sequential(
                nn.Linear(args.htm.dim, 256),
                nn.ReLU(),
                Unflatten(-1, 64, 2, 2),
                nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2), # 11x11
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2), # 22x22
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2), # 45x45
                nn.Tanh(),
                Deprocess_img())
        
    def forward(self, x):
        x = self._net(x)
        # x = rearrange(x, 'b (c t) h w -> b t c h w', c=3)
        # x = x.permute(0, 1, 3, 4, 2)
        x = rearrange(x, 'b c h w -> b h w c')
        return x
    

class CNNSmall(nn.Module):
    def __init__(self, args):
        super(CNNSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, args.hcam.dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        return x


class DeCNNSmall(nn.Module):
    def __init__(self, args):
        super(DeCNNSmall, self).__init__()
        self._net = nn.Sequential(
                nn.Linear(args.hcam.dim, 128),
                nn.ReLU(),
                Unflatten(-1, 32, 2, 2),
                nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=2, stride=2), # 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2), # 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2), # 11x11
                nn.Tanh(),
                Deprocess_img())
        
    def forward(self, x):
        x = self._net(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x