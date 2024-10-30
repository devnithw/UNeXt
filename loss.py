import torch
import torch.nn as nn
import torch.nn.functional as F

# Use DICE from MedPy library
from medpy.metric.binary import dc

class BCEDiceLoss(nn.Module):
    """
    Combination of BCE loss and Dice loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Use PyTorch BCE loss
        BCE = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5 # Avoid division by zero

        # Sigmoid non-linearity
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)

        # Compute intersection
        intersection = (input * target)

        # Compute Dice coefficient (IOU)
        DICE = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        DICE = 1 - DICE.sum() / num

        # Compute combination of BCE + DICE score
        return 0.5 * BCE + DICE

class BCEDiceLossMedPy(nn.Module):
    """
    USing Dice loss from MedPy
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Use PyTorch BCE loss
        BCE = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5 # Avoid division by zero

        # Use MedPy DICE loss
        DICE = dc(input, target)

        # Compute combination of BCE + DICE score
        return 0.5 * BCE + DICE