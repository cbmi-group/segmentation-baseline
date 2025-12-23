import argparse  
import logging  
import os  
import random  
import sys  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torchvision.transforms as transforms  
import torchvision.transforms.functional as TF  
from pathlib import Path  
from torch import optim  
from torch.utils.data import DataLoader, random_split  
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter 
from monai.networks.nets import SwinUNETR
from evaluate import evaluate  
from unet import NestedUNet, UNet, AttU_Net, ResUNet,U2NET
from utils.data_loading import BasicDataset, CarvanaDataset  
from utils.dice_score import dice_loss,DiceLoss  
import logging

dir_img = Path('/dataset/er/img/')  
dir_mask = Path('/dataset/er/mask/')  
dir_checkpoint = Path('./checkpoints/')  

def train_model(  
        model,  
        device,  
        epochs: int = 5,  
        batch_size: int = 1,  
        learning_rate: float = 1e-5,  
        val_percent: float = 0.1,  
        save_checkpoint: bool = True,  
        img_scale: float = 0.5,  
        amp: bool = False,  
        weight_decay: float = 1e-8,  
        momentum: float = 0.999,  
        model_name: str = "UNet"
):  
    # 1. Create dataset  
    try:  
        train_dataset = CarvanaDataset(dir_img, dir_mask, img_scale, mode='train')
        val_dataset = CarvanaDataset(dir_img, dir_mask, img_scale, mode='val')
        # print("CarvanaDataset")  
    except (AssertionError, RuntimeError, IndexError):  
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mode='train')
        val_dataset = BasicDataset(dir_img, dir_mask, img_scale, mode='val')
        # print("BasicDataset")  
    # exit()

    # 2. Split into train / validation partitions  
    n_val = int(len(val_dataset) * val_percent)  
    n_train = len(train_dataset) - n_val  
    train_set = train_dataset
    val_set = val_dataset

    # 3. Create data loaders  
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)  
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=4)  
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)  
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)  

    # Initialize logging  
    writer = SummaryWriter(log_dir=f'runs/{model_name}')  
    logging.info(f'''Starting training:  
        Epochs:          {epochs}  
        Batch size:      {batch_size}  
        Learning rate:   {learning_rate}  
        Training size:   {n_train}  
        Validation size: {n_val}  
        Checkpoints:     {save_checkpoint}  
        Device:          {device.type}  
        Images scaling:  {img_scale}  
        Mixed Precision: {amp}  
    ''')  

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP  
    # optimizer = optim.RMSprop(model.parameters(),  
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)  
    optimizer = optim.AdamW(model.parameters(),
                       lr=learning_rate * 0.2,  
                       weight_decay=1e-4,       
                       betas=(0.9, 0.999))      
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)  # goal: maximize Dice score  
    # optimizer = optim.RMSprop(model.parameters(),  
    #                       lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss() 
    criterion = DiceLoss()
    global_step = 0  

    # 5. Begin training  
    for epoch in range(1, epochs + 1):  
        model.train()  
        epoch_loss = 0  
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:  
            for batch in train_loader:  
                images, true_masks = batch['image'], batch['mask']  
                if model_name!= "SwinUNETR":
                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'  

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)  
                true_masks = true_masks.to(device=device, dtype=torch.long)  

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):  
                    masks_pred = model(images)
                    if args.model_name == "U2Net":
                        loss = 0
                        weights = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  
                        
                        for idx, pred in enumerate(masks_pred):
                            layer_loss = criterion(pred.squeeze(1), true_masks.float())
                            loss += weights[idx] * layer_loss
                        
                    else:
                        # other models
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())  
                    # loss = criterion(masks_pred.squeeze(1), true_masks.float())  
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)  
                if torch.isnan(loss).any():
                    logging.info(f"Epoch {epoch}: Nan loss detected, skipping batch.")
                    optimizer.zero_grad()  
                    continue  # skip this batch
                optimizer.zero_grad(set_to_none=True)  
                grad_scaler.scale(loss).backward()  
                grad_scaler.unscale_(optimizer)  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                grad_scaler.step(optimizer)  
                grad_scaler.update()  

                pbar.update(images.shape[0])  
                global_step += 1  
                epoch_loss += loss.item()  

                # loss -> TensorBoard  
                writer.add_scalar('Loss/train', loss.item(), global_step)  
                pbar.set_postfix(**{'loss (batch)': loss.item()})  

                # Evaluation round  
                division_step = (n_train // (5 * batch_size))  
                if division_step > 0:  
                    if global_step % division_step == 0:  
                        val_score = evaluate(model, val_loader, device, amp)  
                        scheduler.step(val_score)  

                        logging.info('Validation Dice score: {}'.format(val_score))  

                        # dice -> TensorBoard  
                        writer.add_scalar('Dice/validation', val_score, global_step)  

        # record epoch loss
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)  

        if save_checkpoint and epoch % 100 == 0:  
            dir_checkpoint = Path(f'./{model_name}_hard/') 
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()  
            state_dict['mask_values'] = train_dataset.mask_values  
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))  
            logging.info(f'Checkpoint {epoch} saved!')  

    writer.close()  


def get_args():  
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')  
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')  
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')  
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,  
                        help='Learning rate', dest='lr')  
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')  
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')  
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,  
                        help='Percent of the data that is used as validation (0-100)')  
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')  
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')  
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  
    parser.add_argument("--model_name", type=str, required=True, help="name of model")  

    return parser.parse_args()  


if __name__ == '__main__':  
    args = get_args()  

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    logging.info(f'Using device {device}')  

    # Change here to adapt to your data  
    # n_channels=3 for RGB images  
    # n_classes is the number of classifications(just count foreground), 1 represents foreground-background segmentation
    if args.model_name == "UNet":
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == "NestedUNet":
        model = NestedUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == "AttU_Net":
        model = AttU_Net(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == "ResUNet":
        model = ResUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == "U2Net":
        model = U2NET(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == "SwinUNETR":
        model = SwinUNETR(in_channels=1, out_channels=args.classes,img_size = (256,256),feature_size = 48, spatial_dims=2) 
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model = model.to(memory_format=torch.channels_last)  
    if args.model_name != "SwinUNETR":
        logging.info(f'Network:\n'  
                    f'\t{model.n_channels} input channels\n'  
                    f'\t{model.n_classes} output channels (classes)\n'  
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')  

    if args.load:  
        state_dict = torch.load(args.load, map_location=device)  
        del state_dict['mask_values']  
        model.load_state_dict(state_dict)  
        logging.info(f'Model loaded from {args.load}')  

    model.to(device=device)  
    try:  
        train_model(  
            model=model,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.lr,  
            device=device,  
            img_scale=args.scale,  
            val_percent=args.val / 100,  
            amp=args.amp,
            model_name=args.model_name
        )  
    except torch.cuda.OutOfMemoryError:  
        logging.error('Detected OutOfMemoryError! '  
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '  
                      'Consider enabling AMP (--amp) for fast and memory efficient training')  
        torch.cuda.empty_cache()  
        model.use_checkpointing()  
        train_model(  
            model=model,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.lr,  
            device=device,  
            img_scale=args.scale,  
            val_percent=args.val / 100,  
            amp=args.amp,
            model_name=args.model_name  
        )