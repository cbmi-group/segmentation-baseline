import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=True):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            mask_true = mask_true>0.5

            # predict the mask
            mask_pred = net(image)

            # if net.n_classes == 1:
            # tqdm.write(f"mask_true: {mask_true.min()}, {mask_true.max()}") 
            try:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()    
            except AttributeError:  # if # tuple object has no attribute 'sigmoid'
                mask_pred = (F.sigmoid(mask_pred[0]) > 0.5).float()  
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # tqdm.write(f"mask_pred: {mask_pred.min()}, {mask_pred.max()}")
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            
    net.train()
    return dice_score / max(num_val_batches, 1)
