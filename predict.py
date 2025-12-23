import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import tifffile
from utils.data_loading import BasicDataset
from monai.networks.nets import SwinUNETR
from unet import NestedUNet, UNet, AttU_Net, ResUNet, U2NET
from skimage.io import imsave

def min_max_normalize(image):
    img_array = np.array(image)
    min_value = np.min(img_array)
    max_value = np.max(img_array)
    normalized_array = (img_array - min_value) / (max_value - min_value)
    return normalized_array

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    # logging.info("Step 1: Original image unique values: {}".format(np.unique(np.array(full_img))))

    # perform Min-Max normalization on the input image
    full_img = min_max_normalize(full_img)
    # logging.info("Step 2: After Min-Max normalization unique values: {}".format(np.unique(full_img)))

    # turn normalized float array to PIL image
    full_img = Image.fromarray((full_img * 255).astype(np.uint8))

    # crop the image to the center part with size multiple of 256
    cropped_img, new_width, new_height = center_crop_image(full_img)
    # logging.info("Step 3: After center cropping unique values: {}".format(np.unique(np.array(cropped_img))))

    patches = divide_image(cropped_img)
    predicted_patches = []

    for i, patch in enumerate(patches):
        patch_array = np.array(patch).astype(np.float32) / 255.0

        img = torch.from_numpy(patch_array).unsqueeze(0).unsqueeze(0)  
        img = img.to(device=device, dtype=torch.float32)
        # logging.info(f"Step 4: Patch {i} unique values before model: {torch.unique(img)}")

        with torch.no_grad():
            output = net(img)
        
            # process U2Net output if it's a tuple
            if isinstance(output, tuple):
                if all(isinstance(o, torch.Tensor) for o in output):
                    # U2Net usually returns multiple outputs, use the last one as final output
                    output = output[0]
                else:
                    print(f'warning: Model returned a tuple with non-tensor elements. Output structure: {type(output)}')
                    continue  # skip
            
            output = output.cpu()
            # output = net(img).cpu()
            output = F.interpolate(output, (256, 256), mode='bilinear')
            try:
                num_classes = net.out_channels
            except AttributeError:
                # if no out_channels attribute, try n_classes
                try:
                    num_classes = net.n_classes
                except AttributeError:
                    num_classes = 1  
            if num_classes > 1:
                mask = output.argmax(dim=1)
            else:
                # mask = torch.sigmoid(output) > out_threshold
                mask = torch.sigmoid(output)

        # logging.info(f"Step 5: Patch {i} unique values after model prediction: {np.unique(mask.numpy())}")

        # make sure mask is float32
        mask = mask[0].squeeze().numpy().astype(np.float32)  

        # turn mask to uint8
        mask = (mask * 255).astype(np.uint8)  
        mask = Image.fromarray(mask)  
        # logging.info(f"Step 6: Patch {i} unique values after converting to uint8: {np.unique(np.array(mask))}")
        predicted_patches.append(mask)

    # combine patches and record unique values
    result = combine_patches(predicted_patches, (new_width, new_height))
    result = np.array(result).astype(np.float32) / 255.0  
    # logging.info("Step 7: Combined patches unique values: {}".format(np.unique(result)))
    # exit()
    return result

def center_crop_image(image, patch_size=256):
    width, height = image.size
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image, new_width, new_height

def crop_padded_result(result, original_size):
    return result.crop((0, 0, original_size[0], original_size[1]))

def divide_image(image, patch_size=256):
    width, height = image.size
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches

def combine_patches(patches, original_size, patch_size=256):
    width, height = original_size
    result = Image.new('L', original_size)
    index = 0
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = patches[index]
            result.paste(patch, (x, y))
            index += 1
    return result

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Folder path of input images', required=True)
    parser.add_argument('--cropped_img_folder',  type=str ,help='Folder path of cropped img',  required=True)
    parser.add_argument('--cropped_mask_folder',  type=str ,help='Folder path of cropped mask',  required=True)
    parser.add_argument('--mask_folder', type=str ,help='Folder path of mask',  required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Folder path of output masks', required=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def crop(test_folder, mask_folder, cropped_img_folder, cropped_mask_folder, patch_size=256):
    # make sure output folders exist
    os.makedirs(cropped_img_folder, exist_ok=True)
    os.makedirs(cropped_mask_folder, exist_ok=True)
    print(f"test_folder: {test_folder}")
    # traverse test folder images
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.tif')):

            input_path = os.path.join(test_folder, filename)
            cropped_img_path = os.path.join(cropped_img_folder, filename)
            print(cropped_img_path)

            img = Image.open(input_path)
            cropped_img, _, _ = center_crop_image(img, patch_size)
            cropped_img.save(cropped_img_path)
            
            mask_filename = os.path.splitext(filename)[0] + '.tif'
            mask_path = os.path.join(mask_folder, filename)
            # print(mask_filename)
            mask_img = Image.open(mask_path)
            cropped_mask, _, _ = center_crop_image(mask_img, patch_size)
            # print("-----------------")
            # print(mask_path)
            cropped_mask_path = os.path.join(cropped_mask_folder, mask_filename)
            # print("cropped_mask_path:",cropped_mask_path)
            # print("mask_path:",mask_path)

            if os.path.exists(mask_path):
                # print("--------------")
                # print(mask_path)
                mask_img = Image.open(mask_path)
                cropped_mask, _, _ = center_crop_image(mask_img, patch_size)
                cropped_mask.save(cropped_mask_path)
                # saved_image = tifffile.imread(cropped_mask_path)
                # print("shape of saved image:", saved_image.shape)  


            print(f"Processed {filename} and {mask_filename}")
  
    
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_folder = args.input
    output_folder = args.output
    mask_folder = args.mask_folder
    cropped_img_folder = args.cropped_img_folder
    cropped_mask_folder = args.cropped_mask_folder
    patch_size = 256  
    
    os.makedirs(output_folder, exist_ok=True)

    # net = SwinUNETR(
    #     in_channels=1,              # input channels
    #     out_channels=args.classes,  # output classes
    #     img_size=(256, 256),        # image size
    #     feature_size=48,            # feature size (key parameter)
    #     spatial_dims=2              # 2D data
    # )
    net = ResUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.tif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            logging.info(f'Predicting image {input_path} ...')
            img = Image.open(input_path)

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                # save as float TIFF (compatible with ImageJ)
                tifffile.imwrite(
                    output_path, 
                    mask.astype(np.float32),  
                    photometric='minisblack' # single-channel grayscale image
                )
                # mask = mask.astype(np.float16)
                #logging.info(f'Mask saved to {output_path}')
    # print("Processing cropped images and masks...")
    # print(input_folder)
    # print(mask_folder)
    # print(cropped_img_folder)
    # print(cropped_mask_folder)

    # crop the images and masks to patches
    crop(input_folder,mask_folder, cropped_img_folder, cropped_mask_folder, patch_size)


