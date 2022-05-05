import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
    

def module_size(module):
    assert isinstance(module, torch.nn.Module)
    n_params, n_conv_layers = 0, 0
    for name, param in module.named_parameters():
        if 'conv' in name:
            n_conv_layers += 1
        n_params += param.numel()
    return n_params, n_conv_layers

class _ConvLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, out_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv1', nn.Conv3d(in_features, int(out_features/4), kernel_size=3, stride=1, bias=False, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(int(out_features/4)))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(int(out_features/4), int(out_features/2), kernel_size=3, stride=1, bias=False, padding=1))
        self.add_module('norm2', nn.BatchNorm3d(int(out_features/2)))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv3', nn.Conv3d(int(out_features/2), int(out_features), kernel_size=3, stride=1, bias=False, padding=1))
        self.add_module('norm3', nn.BatchNorm3d(out_features))
        self.add_module('relu3', nn.ReLU(inplace=True))
        
    def forward(self, x):
        y = super(_ConvLayer, self).forward(x)
        return y
    
    
class Upsampling3d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, 
            mode='nearest')
    
    
class _UpSampleLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, out_features):
        super(_UpSampleLayer, self).__init__()
        self.add_module('conv', nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, bias=False, padding=1))
        self.add_module('norm', nn.BatchNorm3d(out_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('upsp', Upsampling3d(scale_factor=2))
        
    def forward(self, x):
        y = super(_UpSampleLayer, self).forward(x)
        return y
    
    
class FNet(nn.Module):
    def __init__(self):
        super().__init__() # if without writing this, pytorch cannot assign module before Module.__init__() call
        self.conv_layer_3 = _ConvLayer(in_features=1, out_features=48)
        self.conv_layer_2 = _ConvLayer(in_features=1, out_features=32)
        self.conv_layer_1 = _ConvLayer(in_features=1, out_features=24)
        self.conv_layer_0 = _ConvLayer(in_features=1, out_features=16)
        self.upsp_layer_3 = _UpSampleLayer(in_features=48, out_features=32)
        self.upsp_layer_2 = _UpSampleLayer(in_features=32+32, out_features=24)
        self.upsp_layer_1 = _UpSampleLayer(in_features=24+24, out_features=16)
#         self.upsp_layer_0 = _UpSampleLayer(in_features=16+16, out_features=2)
# the pred and true in torch.nn.BCEWithLogitsLoss are all one channel
        self.conv = nn.Conv3d(in_channels=16+16, out_channels=1, kernel_size=3, stride=1, bias=False, padding=1)
        print('# params {}, # conv layers {}'.format(*self.model_size))
        
    @property
    def model_size(self):
        return module_size(self)
        
    def forward(self, input0, input1, input2, input3):
        cl3 = self.conv_layer_3(input3)
        cl2 = self.conv_layer_2(input2)
        cl1 = self.conv_layer_1(input1)
        cl0 = self.conv_layer_0(input0)
        ul3 = self.upsp_layer_3(cl3)
        ul2 = self.upsp_layer_2(torch.cat([ul3, cl2], dim=1))
        ul1 = self.upsp_layer_1(torch.cat([ul2, cl1], dim=1))
        ul0 = self.conv(torch.cat([ul1, cl0], dim=1))
        
#         output = F.softmax(ul0, dim=1)
        return ul0 # because torch.nn.BCEWithLogitsLoss already incorporates sigmoid function and output should be one channel


class UNet64(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-channel as input
        # nn.Conv3d is a class, contains Conv3d.weight(3w*3h*4c*8) and Conv3d.bias
        self.conv1_8 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv8_8 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv8_16 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv16_16 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv16_32 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv32_32 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv32_64 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv64_64 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv64_32 = nn.Conv3d(64, 32, 3, padding=1)
        self.conv32_16 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv16_8 = nn.Conv3d(16, 8, 3, padding=1)
        self.conv8_1 = nn.Conv3d(8, 1, 3, padding=1)
        # https://zhuanlan.zhihu.com/p/32506912
        self.up_conv64_32 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.up_conv32_16 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.up_conv16_8 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        # print('# params {}, # conv layers {}'.format(*self.model_size))
        
    @property
    def model_size(self):
        return module_size(self)

    def forward(self, input):
        input = input.type('torch.cuda.FloatTensor')
        enc1 = self.conv1_8(input)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()
        enc1 = self.conv8_8(enc1)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()

        enc2 = F.max_pool3d(enc1, 2, 2)
        enc2 = self.conv8_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()
        enc2 = self.conv16_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()

        enc3 = F.max_pool3d(enc2, 2, 2)
        enc3 = self.conv16_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()
        enc3 = self.conv32_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()

        btm = F.max_pool3d(enc3, 2, 2)
        btm = self.conv32_64(btm)
        btm = self.conv64_64(btm)

        dec3 = self.up_conv64_32(btm)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.conv64_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()
        dec3 = self.conv32_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()

        dec2 = self.up_conv32_16(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.conv32_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()
        dec2 = self.conv16_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()

        dec1 = self.up_conv16_8(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.conv16_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()
        dec1 = self.conv8_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()

        output = self.conv8_1(dec1)
        output = torch.sigmoid(output)

        return output



class UNet256(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-channel as input
        # nn.Conv3d is a class, contains Conv3d.weight(3w*3h*4c*8) and Conv3d.bias
        self.conv1_8 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv8_8 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv8_16 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv16_16 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv16_32 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv32_32 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv32_64 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv64_64 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv64_128 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv128_128 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv128_256 = nn.Conv3d(128, 256, 3, padding=1)

        self.conv256_256 = nn.Conv3d(256, 256, 3, padding=1)

        self.conv256_128 = nn.Conv3d(256, 128, 3, padding=1)
        self.conv128_64 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv64_32 = nn.Conv3d(64, 32, 3, padding=1)
        self.conv32_16 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv16_8 = nn.Conv3d(16, 8, 3, padding=1)
        self.conv8_1 = nn.Conv3d(8, 1, 3, padding=1)
        # https://zhuanlan.zhihu.com/p/32506912
        self.up_conv256_128 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.up_conv128_64 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_conv64_32 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.up_conv32_16 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.up_conv16_8 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        print('# params {}, # conv layers {}'.format(*self.model_size))
        
    @property
    def model_size(self):
        return module_size(self)

    def forward(self, input):
        input = input.type('torch.cuda.DoubleTensor')
        enc1 = self.conv1_8(input)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()
        enc1 = self.conv8_8(enc1)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()

        enc2 = F.max_pool3d(enc1, 2, 2)
        enc2 = self.conv8_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()
        enc2 = self.conv16_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()

        enc3 = F.max_pool3d(enc2, 2, 2)
        enc3 = self.conv16_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()
        enc3 = self.conv32_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()

        enc4 = F.max_pool3d(enc3, 2, 2)
        enc4 = self.conv32_64(enc4)
        enc4 = F.instance_norm(enc4)
        nn.LeakyReLU()
        enc4 = self.conv64_64(enc4)
        enc4 = F.instance_norm(enc4)
        nn.LeakyReLU()

        enc5 = F.max_pool3d(enc4, 2, 2)
        enc5 = self.conv64_128(enc5)
        enc5 = F.instance_norm(enc5)
        nn.LeakyReLU()
        enc5 = self.conv128_128(enc5)
        enc5 = F.instance_norm(enc5)
        nn.LeakyReLU()

        btm = F.max_pool3d(enc5, 2, 2)
        btm = self.conv128_256(btm)
        btm = self.conv256_256(btm)

        dec5 = self.up_conv256_128(btm)
        dec5 = torch.cat([enc5, dec5], dim=1)
        dec5 = self.conv256_128(dec5)
        dec5 = F.instance_norm(dec5)
        nn.LeakyReLU()
        dec5 = self.conv128_128(dec5)
        dec5 = F.instance_norm(dec5)
        nn.LeakyReLU()

        dec4 = self.up_conv128_64(dec5)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.conv128_64(dec4)
        dec4 = F.instance_norm(dec4)
        nn.LeakyReLU()
        dec4 = self.conv64_64(dec4)
        dec4 = F.instance_norm(dec4)
        nn.LeakyReLU()

        dec3 = self.up_conv64_32(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.conv64_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()
        dec3 = self.conv32_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()

        dec2 = self.up_conv32_16(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.conv32_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()
        dec2 = self.conv16_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()

        dec1 = self.up_conv16_8(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.conv16_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()
        dec1 = self.conv8_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()

        output = self.conv8_1(dec1)
        # output = F.softmax(output, dim=1)

        return output

class UNet1024(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-channel as input
        # nn.Conv3d is a class, contains Conv3d.weight(3w*3h*4c*8) and Conv3d.bias
        self.conv1_8 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv8_8 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv8_16 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv16_16 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv16_32 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv32_32 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv32_64 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv64_64 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv64_128 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv128_128 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv128_256 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv256_256 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv256_512 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv512_512 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv512_1024 = nn.Conv3d(512, 1024, 3, padding=1)

        self.conv1024_1024 = nn.Conv3d(1024, 1024, 3, padding=1)

        self.conv1024_512 = nn.Conv3d(1024, 512, 3, padding=1)
        self.conv512_256 = nn.Conv3d(512, 256, 3, padding=1)
        self.conv256_128 = nn.Conv3d(256, 128, 3, padding=1)
        self.conv128_64 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv64_32 = nn.Conv3d(64, 32, 3, padding=1)
        self.conv32_16 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv16_8 = nn.Conv3d(16, 8, 3, padding=1)
        self.conv8_1 = nn.Conv3d(8, 1, 3, padding=1)
        # https://zhuanlan.zhihu.com/p/32506912
        self.up_conv1024_512 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up_conv512_256 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up_conv256_128 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.up_conv128_64 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_conv64_32 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.up_conv32_16 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.up_conv16_8 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        print('# params {}, # conv layers {}'.format(*self.model_size))
        
    @property
    def model_size(self):
        return module_size(self)

    def forward(self, input):
        input = input.type('torch.cuda.DoubleTensor')
        enc1 = self.conv1_8(input)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()
        enc1 = self.conv8_8(enc1)
        enc1 = F.instance_norm(enc1)
        nn.LeakyReLU()

        enc2 = F.max_pool3d(enc1, 2, 2)
        enc2 = self.conv8_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()
        enc2 = self.conv16_16(enc2)
        enc2 = F.instance_norm(enc2)
        nn.LeakyReLU()

        enc3 = F.max_pool3d(enc2, 2, 2)
        enc3 = self.conv16_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()
        enc3 = self.conv32_32(enc3)
        enc3 = F.instance_norm(enc3)
        nn.LeakyReLU()

        enc4 = F.max_pool3d(enc3, 2, 2)
        enc4 = self.conv32_64(enc4)
        enc4 = F.instance_norm(enc4)
        nn.LeakyReLU()
        enc4 = self.conv64_64(enc4)
        enc4 = F.instance_norm(enc4)
        nn.LeakyReLU()

        enc5 = F.max_pool3d(enc4, 2, 2)
        enc5 = self.conv64_128(enc5)
        enc5 = F.instance_norm(enc5)
        nn.LeakyReLU()
        enc5 = self.conv128_128(enc5)
        enc5 = F.instance_norm(enc5)
        nn.LeakyReLU()

        enc6 = F.max_pool3d(enc5, 2, 2)
        enc6 = self.conv128_256(enc6)
        enc6 = F.instance_norm(enc6)
        nn.LeakyReLU()
        enc6 = self.conv256_256(enc6)
        enc6 = F.instance_norm(enc6)
        nn.LeakyReLU()

        enc7 = F.max_pool3d(enc6, 2, 2)
        enc7 = self.conv256_512(enc7)
        enc7 = F.instance_norm(enc7)
        nn.LeakyReLU()
        enc7 = self.conv512_512(enc7)
        enc7 = F.instance_norm(enc7)
        nn.LeakyReLU()

        btm = F.max_pool3d(enc7, 2, 2)
        btm = self.conv512_1024(btm)
        btm = self.conv1024_1024(btm)

        dec7 = self.up_conv1024_512(btm)
        dec7 = torch.cat([enc7, dec7], dim=1)
        dec7 = self.conv1024_512(dec7)
        dec7 = F.instance_norm(dec7)
        nn.LeakyReLU()
        dec7 = self.conv512_512(dec7)
        dec7 = F.instance_norm(dec7)
        nn.LeakyReLU()

        dec6 = self.up_conv512_256(dec7)
        dec6 = torch.cat([enc6, dec6], dim=1)
        dec6 = self.conv512_256(dec6)
        dec6 = F.instance_norm(dec6)
        nn.LeakyReLU()
        dec6 = self.conv256_256(dec6)
        dec6 = F.instance_norm(dec6)
        nn.LeakyReLU()

        dec5 = self.up_conv256_128(dec6)
        dec5 = torch.cat([enc5, dec5], dim=1)
        dec5 = self.conv256_128(dec5)
        dec5 = F.instance_norm(dec5)
        nn.LeakyReLU()
        dec5 = self.conv128_128(dec5)
        dec5 = F.instance_norm(dec5)
        nn.LeakyReLU()

        dec4 = self.up_conv128_64(dec5)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.conv128_64(dec4)
        dec4 = F.instance_norm(dec4)
        nn.LeakyReLU()
        dec4 = self.conv64_64(dec4)
        dec4 = F.instance_norm(dec4)
        nn.LeakyReLU()

        dec3 = self.up_conv64_32(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.conv64_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()
        dec3 = self.conv32_32(dec3)
        dec3 = F.instance_norm(dec3)
        nn.LeakyReLU()

        dec2 = self.up_conv32_16(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.conv32_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()
        dec2 = self.conv16_16(dec2)
        dec2 = F.instance_norm(dec2)
        nn.LeakyReLU()

        dec1 = self.up_conv16_8(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.conv16_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()
        dec1 = self.conv8_8(dec1)
        dec1 = F.instance_norm(dec1)
        nn.LeakyReLU()

        output = self.conv8_1(dec1)
        # output = F.softmax(output, dim=1)

        return output

