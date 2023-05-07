import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from scipy import ndimage

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from Models.layers.channel_attention_layer import SE_Conv_Block
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.layers.nonlocal_layer import NONLocalBlock2D


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size
        self.counter = 0

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv3 = conv_block(self.in_channels, filters[0])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.conv4 = conv_block(filters[0], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=8, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv3 = self.conv3(inputs)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)
        up4, att_weight4 = self.up4(g_conv4)
        
        upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        up4 = upsample(up4)
        
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv_cat = torch.cat([dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)
        
        self.counter += 1

        return out