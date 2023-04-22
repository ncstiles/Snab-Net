
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
        self.maxpool1 = nn.MaxPool2d(kernel_size=(16, 16))


        self.center = conv_block(filters[0], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[4],
                                                    inter_channels=filters[0])
        
        # upsampling
        self.up_concat1 = UpCat(filters[4], filters[0], self.is_deconv)

        self.up1 = SE_Conv_Block(filters[1], filters[0], drop_out=True)

        # deep supervision
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=4, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        img = inputs[0]
        save_image(img, f'./result/original_img_{self.counter}.jpg')
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        # Gating Signal Generation
        center = self.center(maxpool1)

        # Attention Mechanism
        # Upscaling Part (Decoder)

        g_conv1, att1 = self.attentionblock1(conv1, center)
        upsample = nn.Upsample(scale_factor=8, mode="bilinear")
        up_center = upsample(center)
        up1 = self.up_concat1(conv1, up_center)
        up1, att_weight1 = self.up1(up1)

        atten1_map = att1.cpu().detach().numpy().astype(float)
        atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
                                                             300 / atten1_map.shape[3]], order=0)
        
        atten1_map_to_save = atten1_map[0, 0, :, :].reshape(224, 300)
        rescaled = (255.0 / (atten1_map_to_save.max() - atten1_map_to_save.min()) * (atten1_map_to_save - atten1_map_to_save.min())).astype(np.uint8)
        img = Image.fromarray(rescaled).convert('RGB')
        img.save(f'./result/attention_map_{self.counter}.jpg')
        
        # Deep Supervision
        dsv1 = self.dsv1(up1)
        
        dsv_cat = torch.cat([dsv1], dim=1)

        out = self.scale_att(dsv_cat)

        out = self.final(out)
        
        self.counter += 1

        return out