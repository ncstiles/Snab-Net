
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
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1], drop_out=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(8, 8))

        self.center = conv_block(filters[1], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
                                                    inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[4], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        

        # upsampling
        self.up_concat2 = UpCat(filters[4], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)

        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=8, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        img = inputs[0]
        save_image(img, f'./result/original_img_{self.counter}.jpg')
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # Gating Signal Generation
        center = self.center(maxpool2)

        # Attention Mechanism
        # Upscaling Part (Decoder)

        g_conv2, att2 = self.attentionblock2(conv2, center)

        upsample = nn.Upsample(scale_factor=4, mode="bilinear")
        up_center = upsample(center)
        up2 = self.up_concat2(g_conv2, up_center)
        up2, att_weight2 = self.up2(up2)
        g_conv1, att1 = self.attentionblock1(conv1, up2)

        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        dsv_cat = torch.cat([dsv1, dsv2], dim=1)

        out = self.scale_att(dsv_cat)

        out = self.final(out)
        
        self.counter += 1

        return out
