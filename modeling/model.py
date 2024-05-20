import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Traditional Convolution
class TC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TC, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, input):
        out = self.conv(input)
        return out



# Depthwise Separable Convolution
class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class enhance_net_nopool(nn.Module):

    def __init__(self, scale_factor, conv_type='dsc'):
        print("model")
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # Define Conv type
        if conv_type == 'dsc':
            self.conv = DSC
        elif conv_type == 'dc':
            self.conv = DC
        elif conv_type == 'tc':
            self.conv = TC
        else:
            print("conv type is not available")

        #   zerodce DWC + p-shared
        self.e_conv1 = self.conv(3, number_f)

        self.e_conv2 = self.conv(number_f, number_f)
        self.e_conv3 = self.conv(number_f, number_f)
        self.e_conv4 = self.conv(number_f, number_f)

        self.e_conv5 = self.conv(number_f * 2, number_f)
        self.e_conv6 = self.conv(number_f * 2, number_f)
        self.e_conv7 = self.conv(number_f , number_f)
        self.e_conv8 = self.conv(number_f * 2 , number_f)

        self.e_conv9 = self.conv(number_f * 2, 3)
        self.e_conv10 = self.conv(number_f * 2 , 4)

    def enhance(self, x, x_r):
        x_r,w = x_r[0],x_r[1]
        w= w = torch.mean(w,dim=[0,2,3])
        w1,w2,w3,w4 = torch.split(w,1,dim=0)
        x1 = x + x_r * (torch.pow(x, 2) - x)
        x2= x1 + x_r * (torch.pow(x1, 2) - x1 - pow(w1,3)*x)
        
        x3 = x2 + x_r * (torch.pow(x2, 2) - x2 -pow(w2,3)*x1 -pow(w2,5)*x)
        enhance_image_1 = x3 + x_r * (torch.pow(x3, 2) - x3 - pow(w2,3)*x2 -pow(w2,5)*x1 -pow(w2,7)*x)

        x4 = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1 - pow(w3,3)*x3-pow(w3,5)*x2-pow(w3,7)*x1)
        x5 = x4 + x_r * (torch.pow(x4, 2) - x4 - pow(w3,3)*enhance_image_1 -pow(w3,5)*x3-pow(w3,7)*x2)
        
        x6 = x5 + x_r * (torch.pow(x5, 2) - x5 - pow(w4,3)*x4 -pow(w4,5)*enhance_image_1-pow(w4,7)*x3)
        enhance_image = x6 + x_r * (torch.pow(x6, 2) - x6 - pow(w4,3)*x5-pow(w4,5)*x4-pow(w4,7)*enhance_image_1)
 
        return enhance_image


    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        # extraction
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x7 = self.relu(self.e_conv7(x6))
        x8 = self.relu(self.e_conv8(torch.cat([x5, x7], 1)))
        x_r = F.tanh(self.e_conv9(torch.cat([x1, x8], 1)))
        
        w = F.sigmoid(self.e_conv10(torch.cat([x3,x8], 1)))
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
            w = self.upsample(w)

        # enhancement
        enhance_image = self.enhance(x, [x_r,w])

        return enhance_image, x_r

'''
        # This is for vis ONLY
        x_r = -3 * x_r.squeeze(0).permute(1, 2, 0)
        plt.imshow(x_r)
        plt.show()
        plt.close()

        plt.savefig('test.svg')

        sys.exit(0)
'''



