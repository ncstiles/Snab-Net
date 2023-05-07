import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from scipy import ndimage
import matplotlib.pyplot as plt
from datetime import datetime

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


def visualize_map(attn_map_3, attn_map_2, attn_map_1, ground_img):
    num_maps = 0
    num_maps = num_maps + 1 if attn_map_3 != None else num_maps
    num_maps = num_maps + 1 if attn_map_2 != None else num_maps
    num_maps = num_maps + 1 if attn_map_1 != None else num_maps
    fig, axs = plt.subplots(num_maps, 2)
    print("number of maps:", num_maps)

    # choose random image from batch to visualize

    smallest_batchsize = min(attn_map.shape[0] for attn_map in [attn_map_3, attn_map_2, attn_map_1] if attn_map != None)

    for batch_ix in range(smallest_batchsize):
        plt_ix = 0
        if attn_map_3 != None:
        # visualize attention block 3
            map = attn_map_3.cpu().detach().numpy().astype(float)
            reshaped_map, channels = scale_and_reorder_image(map)

            for i in range(channels):
                title = f"AB 3 channel {i}"
                axs[plt_ix, i].imshow(reshaped_map[batch_ix, :, :, i])
                axs[plt_ix, i].set_title(title)
            
            plt_ix += 1

        if attn_map_2 != None:
            # visualize attention block 2
            map = attn_map_2.cpu().detach().numpy().astype(float)
            reshaped_map, channels = scale_and_reorder_image(map)

            for i in range(channels):
                title = f"AB 2 channel {i}"
                axs[plt_ix, i].imshow(reshaped_map[batch_ix, :, :, i])
                axs[plt_ix, i].set_title(title)
            
            plt_ix += 1

        if attn_map_1 != None:
            # vizsualize attention block 1
            map = attn_map_1.cpu().detach().numpy().astype(float)
            reshaped_map, channels = scale_and_reorder_image(map)

            if num_maps == 1:
                axs[0].imshow(reshaped_map[batch_ix, :, :, 0])
                axs[0].set_title(f"AB 1 channel 0")
            else:
                for i in range(channels):
                    title = f"AB 1 channel {i}"
                    axs[plt_ix, i].imshow(reshaped_map[batch_ix, :, :, i])
                    axs[plt_ix , i].set_title(title)


        # visualize ground truth
        map = np.copy(ground_img.cpu().detach()).astype(float)
        ground_map, channels = scale_and_reorder_image(map)
        if num_maps == 1:
            axs[1].imshow(ground_map[batch_ix, :, :, :])
            axs[1].set_title("Original image")
        else:
            axs[plt_ix, 1].imshow(ground_map[batch_ix, :, :, :])
            axs[plt_ix, 1].set_title("Original image")

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        fig.tight_layout()

        # get current time
        timestamp = datetime.now().strftime("%m_%d:%H_%M_%S.%f")
        plt.savefig(f"attn_out/ablate_23_{timestamp}.png")

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
        self.maxpool1 = nn.MaxPool2d(kernel_size=(8, 8))

        self.conv4 = conv_block(filters[0], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[3],
                                                    inter_channels=filters[0])
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat1 = UpCat(filters[3], filters[0], self.is_deconv)
        
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=8, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs, viz):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        
        conv4 = self.conv4(maxpool1)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)
        up4, att_weight4 = self.up4(g_conv4)
        
        g_conv1, att1 = self.attentionblock1(conv1, up4)

        upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        up4 = upsample(up4)
        
        up1 = self.up_concat1(conv1, up4)
        up1, att_weight1 = self.up1(up1)

        if viz: 
            print("visualizing")
            visualize_map(None, None, att1, inputs)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv4], dim=1)
   
        out = self.scale_att(dsv_cat)
        out = self.final(out)
        
        self.counter += 1

        return out
