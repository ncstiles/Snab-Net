import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_score = torch.mean(-torch.log(dice_score))
    return dice_score


def val_dice_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)
    placenta_dice = dice_score[1]
    brain_dice = dice_score[2]

    return placenta_dice, brain_dice


def Intersection_over_Union_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    dice_mean_score = torch.mean(iou_score)
    placenta_iou = iou_score[1]
    brain_iou = iou_score[2]

    return placenta_iou, brain_iou

def is_in_bounds(point, rows, cols):
    p1, p2 = point
    if p1 < 0 or p1 >= rows:
        return False
    elif p2 < 0 or p2 >= cols:
        return False
    return True

def detect_edges(image):
        rows = int(image.shape[0])
        cols = int(image.shape[1])
        edges = np.zeros(image.shape)
        edges_adj = np.zeros(image.shape)

        for r in range(rows):
            for c in range(cols):
                neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]

                if image[r, c] == 0:
                    continue

                for neighbor in neighbors:
                    nr, nc = neighbor
                    if not is_in_bounds(neighbor, rows, cols):
                        continue

                    neighbor_val = image[nr, nc]
                    if neighbor_val == 0:
                        edges[r, c] = 1

                if edges[r, c] == 1:
                    edges_adj[r, c] = 1
                    for neighbor in neighbors:
                        if is_in_bounds(neighbor, rows, cols):
                            edges_adj[neighbor[0], neighbor[1]] = 1

        return edges_adj

def draw_edge(image, mask, channel):
    for c in range(3):
        color_channel = image[:, :, c]
        if c == channel:
            color_channel[mask == 1] = 1
        else:
            color_channel[mask == 1] = 0
        image[:, :, c] = color_channel

    return image

def val_dice_isic(prediction, soft_ground_truth, num_class, mode=None, image=None):
    pred = prediction.contiguous().view(-1, num_class)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)

    if mode == "test":
        ground_np = ground.cpu().detach().numpy().astype(int)
        pred_np = pred.cpu().detach().numpy().astype(int)
        image_np = np.copy(image.cpu().detach())
        image2show = np.transpose(image_np, (0, 2, 3, 1))

        print("ground np", ground_np.shape)
        print("pred np:", pred_np.shape)
        print("image2show:", image2show.shape)

        for i in range(image2show.shape[0]):
            IMG_SZ = 224*300

            sns.set()
            sns.set_style("whitegrid", {'axes.grid' : False})
            rc = {"axes.spines.left" : False,
                "axes.spines.right" : False,
                "axes.spines.bottom" : False,
                "axes.spines.top" : False,
                "xtick.bottom" : False,
                "xtick.labelbottom" : False,
                "ytick.labelleft" : False,
                "ytick.left" : False,
                "axes.grid" : False}
            plt.rcParams.update(rc)

            ground2show = detect_edges(ground_np[IMG_SZ*i : IMG_SZ*(i+1), 1].reshape(224, 300))
            pred2show = detect_edges(pred_np[IMG_SZ*i : IMG_SZ*(i+1), 1].reshape(224, 300))
            
            im = image2show[i, :, :, :]
            img_w_ground = draw_edge(im, ground2show, 1)
            img_combined = draw_edge(img_w_ground, pred2show, 2)
            plot = plt.imshow(img_combined)
            timestamp = datetime.now().strftime("%m_%d_%H:%M:%S.%f")
            plot.figure.savefig(f"segmentation/ablate_13_{timestamp}.png")
            plt.clf()

    return dice_mean_score


def Intersection_over_Union_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score

