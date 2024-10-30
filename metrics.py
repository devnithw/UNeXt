import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    """
    Computes the intersection over union (IoU) score.
    """
    smooth = 1e-5 # Avoid division by zero

    # Bring data to CPU
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # Binarize the output
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    """
    Computes the Dice coefficient.
    """
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth) # Reformatted as fraction