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


def f1_score(output, target, threshold=0.5):
    """Calculate the F1 score for pixel-wise classification."""
    output = (torch.sigmoid(output) > threshold).float()  # Apply threshold to output
    target = target.float()

    tp = (output * target).sum()  # True positives
    fp = (output * (1 - target)).sum()  # False positives
    fn = ((1 - output) * target).sum()  # False negatives

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return f1.item()