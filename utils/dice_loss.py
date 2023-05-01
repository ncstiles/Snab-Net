import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



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


def val_dice_isic(prediction, soft_ground_truth, num_class, mode=None):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)
    
    if mode == "test":
        ground_np = ground.cpu().detach().numpy().astype(int)
        pred_np = pred.cpu().detach().numpy().astype(int)
        
        for i in range(10):
            OUTPUT_SIZE = 224*300
            ground_to_show_1 = ground_np[OUTPUT_SIZE*i:OUTPUT_SIZE*(i+1), 0].reshape(224, 300)
            ground_to_show_2 = ground_np[OUTPUT_SIZE*i:OUTPUT_SIZE*(i+1), 1].reshape(224, 300)
            pred_to_show_1 = pred_np[OUTPUT_SIZE*i:OUTPUT_SIZE*(i+1), 0].reshape(224, 300)
            pred_to_show_2 = pred_np[OUTPUT_SIZE*i:OUTPUT_SIZE*(i+1), 1].reshape(224, 300)

            sns.set()
            
            plot1 = sns.heatmap(ground_to_show_1)
            plot1.figure.savefig("vis/image_" + str(i) + "_ground1.jpg")
            plt.clf()
            
            plot2 = sns.heatmap(ground_to_show_2)
            plot2.figure.savefig("vis/image_" + str(i) + "_ground2.jpg")
            plt.clf()
            
            plot3 = sns.heatmap(pred_to_show_1)
            plot3.figure.savefig("vis/image_" + str(i) + "_pred1.jpg")
            plt.clf()
            
            plot4 = sns.heatmap(pred_to_show_2)
            plot4.figure.savefig("vis/image_" + str(i) + "_pred2.jpg")
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
