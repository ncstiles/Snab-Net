import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from scipy import ndimage
import matplotlib.pyplot as plt
from datetime import datetime
import random

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from Models.layers.channel_attention_layer import SE_Conv_Block
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.layers.nonlocal_layer import NONLocalBlock2D


def scale_and_reorder_image(attention_map):
    """
    attention map is in form [batch_size, num_channels, height, width]
    reshape to [batch_size, height, width, num_channels]
    make the image (224, 300)
    """
    batch_scale, channel_scale = 1.0, 1.0
    desired_height, desired_width = 224, 300

    batch_size, num_channels, height, width = attention_map.shape
    height_scale, width_scale = desired_height / height, desired_width / width

    zoomed_map = ndimage.interpolation.zoom(attention_map, [batch_scale, channel_scale, height_scale, width_scale], order=0)

    atten_reshaped = np.transpose(zoomed_map, (0,2,3,1))

    return atten_reshaped, atten_reshaped.shape[3]

def visualize_map(attn_map_3, attn_map_2, attn_map_1, scale_map, ground_img):
    fig, axs = plt.subplots(5, 2)


    # choose random image from batch to visualize
    smallest_batchsize = min(attn_map_1.shape[0], attn_map_2.shape[0], attn_map_3.shape[0], scale_map.shape[0])

    for batch_ix in range(smallest_batchsize):
        # visualize attention block 3
        map = attn_map_3.cpu().detach().numpy().astype(float)
        reshaped_map, channels = scale_and_reorder_image(map)

        for i in range(channels):
            title = f"AB 3 channel {i}"
            axs[0, i].imshow(reshaped_map[batch_ix, :, :, i])
            axs[0, i].set_title(title)

        # visualize attention block 2
        map = attn_map_2.cpu().detach().numpy().astype(float)
        reshaped_map, channels = scale_and_reorder_image(map)

        for i in range(channels):
            title = f"AB 2 channel {i}"
            axs[1, i].imshow(reshaped_map[batch_ix, :, :, i])
            axs[1, i].set_title(title)

        # visualize attention block 1
        map = attn_map_1.cpu().detach().numpy().astype(float)
        reshaped_map, channels = scale_and_reorder_image(map)

        for i in range(channels):
            title = f"AB 1 channel {i}"
            axs[2, i].imshow(reshaped_map[batch_ix, :, :, i])
            axs[2, i].set_title(title)

        # visualize ground truth
        map = ground_img.cpu().detach().numpy().astype(float)
        ground_map, _ = scale_and_reorder_image(map)

        axs[2, 1].imshow(ground_map[batch_ix, :, :, :])
        axs[2, 1].set_title("Original image")

        # # visualize the 4 different channels of scale attention
        # map = scale_map.cpu().detach().numpy().astype(float)
        # reshaped_map, channels = scale_and_reorder_image(map)

        # # for i in range(0, channels, 4):
        # #     title = f"Attention block 0 channel {i // 4}"
        # axs[3, 0].imshow(reshaped_map[batch_ix, :, :, 0])
        # axs[3, 1].imshow(reshaped_map[batch_ix, :, :, 4])
        # axs[4, 0].imshow(reshaped_map[batch_ix, :, :, 8])
        # axs[4, 1].imshow(reshaped_map[batch_ix, :, :, 12])


        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        fig.tight_layout()


        # get current time
        timestamp = datetime.now().strftime("%m_%d_%H:%M:%S.%f")
        plt.savefig(f"attn_out/{timestamp}.png")

def viz_scale(scale_attn, scale_attn_soft):
    fig, axs = plt.subplots(4, 6)

    # choose random image from batch to visualize

        # visualize attention block 3
    map = scale_attn.cpu().detach().numpy().astype(float)
    reshaped_map, channels = scale_and_reorder_image(map)

    map2 = scale_attn_soft.cpu().detach().numpy().astype(float)
    reshaped_map2, channels = scale_and_reorder_image(map)

    for i in range(channels):
        # title = f"Scale attent {i}"
        axs[i//4, i%4].imshow(reshaped_map[0, :, :, i])
        # axs[i//4, i].set_title(title)

    for i in range(8):
        axs[i//2, 4 + i%2].imshow(reshaped_map[0, :, :, i])


    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()

    plt.show()

        # # get current time
        # timestamp = datetime.now().strftime("%m_%d_%H:%M:%S.%f")
        # plt.savefig(f"attn_out/{timestamp}.png")


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1), viz=False):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
                                                    inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs, viz):

        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
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
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
        g_conv1, att1 = self.attentionblock1(conv1, up2)

        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out, att0, att0soft = self.scale_att(dsv_cat)

        if viz: 
            print("visualizing")
            visualize_map(att3, att2, att1, att0, inputs)




        out = self.final(out)

        return out