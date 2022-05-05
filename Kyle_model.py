import torch
from torch import nn

class Horizontal(torch.nn.Module):
    def __init__(self, num_in, num_out):
        super(Horizontal, self).__init__()
        self.horiz = torch.nn.Sequential(
        
            torch.nn.Conv3d(num_in, num_out, 3, padding=1), torch.nn.BatchNorm3d(num_out), torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(num_out, num_out, 3, padding=1), torch.nn.BatchNorm3d(num_out), torch.nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.horiz(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.horiz_1 = Horizontal(1, 64)
        self.horiz_2 = Horizontal(64, 128)
        self.horiz_3 = Horizontal(128, 256)
        self.horiz_4 = Horizontal(256, 512)
        
        self.downsample = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.horiz_5 = Horizontal(256+512, 256)
        self.horiz_6 = Horizontal(128+256, 128)
        self.horiz_7 = Horizontal(64+128, 64)
        
        self.up1 = torch.nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.up2 = torch.nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.up3 = torch.nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        
        self.one_by_one_conv = torch.nn.Conv3d(64, 1, 1, padding=0)
            

    def forward(self, x):
        # Contracting Path
        x1 = self.horiz_1(x)
        x2 = self.downsample(x1)
        x3 = self.horiz_2(x2)
        x4 = self.downsample(x3)
        x5 = self.horiz_3(x4)
        x6 = self.downsample(x5)
        x7 = self.horiz_4(x6)
        
        # Expansive Path
        x8 = torch.cat((self.up1(x7), x5), axis=1)
        x9 = self.horiz_5(x8)
        x10 = torch.cat((self.up2(x9), x3), axis=1)
        x11 = self.horiz_6(x10)
        x12 = torch.cat((self.up3(x11), x1), axis=1)
        x13 = self.horiz_7(x12)

        out = self.one_by_one_conv(x13)
        
        return torch.sigmoid(out)