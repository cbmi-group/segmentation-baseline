import torch
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
from typing import Callable

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single target
    # tqdm.write(f"input&target: {input.size()}, {target.size()}")  
    # tqdm.write(f"input: {input.size()}, dtype: {input.dtype}, device: {input.device}")  
    # tqdm.write(f"target: {target.size()}, dtype: {target.dtype}, device: {target.device}")
    target = target.to(dtype=input.dtype)
    input = input.squeeze()  
    target = target.squeeze()
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class DiceLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.sigmoid(input)
        target = target.float()

        # calculate dice loss
        intersection = (input * target).sum()
        union = input.sum() + target.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)  
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


