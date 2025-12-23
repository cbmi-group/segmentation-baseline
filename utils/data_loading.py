import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import random
import torch


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', mode: str = 'train'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.mode = mode

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:  
            return img.astype(np.float32)
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            img = (img - img.min()) / (img.max() - img.min())

            return img

    @staticmethod
    def random_crop(img, mask, crop_size=(256, 256)):
        # logging.info(f'shape of img: {img.shape}')
        assert img.shape[1] >= crop_size[0] and img.shape[2] >= crop_size[1], 'Crop size is larger than the image size'
        x = random.randint(0, img.shape[1] - crop_size[0])
        y = random.randint(0, img.shape[2] - crop_size[1])
        img = img[:, x:x+crop_size[0], y:y+crop_size[1]]
        mask = mask[x:x+crop_size[0], y:y+crop_size[1]]
        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # turn mask into binary mask
        mask = np.asarray(mask)  

        if np.any(mask > 1):

            mask_max = np.max(mask)
            mask = mask > mask_max*0.5 
            # mask = (mask > 0.5).astype(np.uint8)  
            mask = Image.fromarray(mask)
            # logging.info(f"Mask for {name} has been binarized.")
            # exit()
        else:
            mask_max = np.max(mask)
            mask = mask > mask_max*0.5 
            mask = Image.fromarray(mask)
        img_array = np.array(img)

        # check image channels
        if len(img_array.shape) == 2:  
            img_array = img_array.astype(np.float32) 
            if img_array.max() - img_array.min() == 0:
                img_array = np.zeros_like(img_array)  
            else:
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())  # Min-Max normalization
            img = Image.fromarray((img_array * 255).astype(np.uint8))  
        elif len(img_array.shape) == 3:  
            img_array = img_array / 255.0  # normalization to [0, 1]
            img = Image.fromarray((img_array * 255).astype(np.uint8))  

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        img, mask = self.random_crop(img, mask)
        img_unique = np.unique(img)
        mask_unique = np.unique(mask)
        # logging.info(f"Image unique values: {img_unique}, count: {len(img_unique)}")
        # logging.info(f"Mask unique values: {mask_unique}, count: {len(mask_unique)}")

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, mode='train'):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='', mode=mode)
