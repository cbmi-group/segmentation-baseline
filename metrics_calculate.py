from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os
from PIL import Image
from medpy.metric.binary import hd95, assd
import albumentations as A
import SimpleITK as sitk
import imageio.v3 as iio
from PIL import Image
from skimage.metrics import structural_similarity as ssim 

def mask2edge(mask):
    """单像素边缘提取 (基于形态学腐蚀)[8](@ref)"""
    eroded = binary_erosion(mask, structure=np.ones((3,3)))
    return mask ^ eroded  # XOR操作获取边缘

def compute_fom(pred_edge, gt_edge, alpha=1/9):
    """优点图计算 (优化距离加权公式)[1](@ref)"""
    if np.count_nonzero(pred_edge)*np.count_nonzero(gt_edge) == 0:
        return 0.0  # 空边缘保护
    
    dist_map = distance_transform_edt(np.logical_not(gt_edge))  # 真实边缘的距离变换
    numerator = np.sum(1 / (1 + alpha * (dist_map[pred_edge]**2)))
    denominator = max(np.sum(pred_edge), np.sum(gt_edge))  # 动态归一化
    return numerator / denominator

def compute_tv(binary_img):
    """各向同性全变分计算 (适用于二值图像)[8](@ref)"""
    dx = np.abs(np.gradient(binary_img.astype(float), axis=0))
    dy = np.abs(np.gradient(binary_img.astype(float), axis=1))
    return np.sqrt(dx**2 + dy**2).sum()

def eval_distance(mask_list, seg_result_list, num_classes, image_names=None):
    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2
    assert len(mask_list) == len(seg_result_list)
    
    if num_classes == 2:
        hd_list = []
        sd_list = []
        fom_list = []  # FOM storage
        max_hd = -1
        max_hd_image = []
        # save abnormal image names
        hd_image_names = []
        for i in range(len(mask_list)):
            pred = seg_result_list[i]
            gt = mask_list[i]

            if np.any(pred) and np.any(gt):
                # calculate HD95 and ASSD
                hd_ = hd95(pred, gt)
                sd_ = assd(pred, gt)
                hd_list.append(hd_)
                sd_list.append(sd_)
                
                # track large HD95 and corresponding image name
                # if hd_ > 20:
                #     print(f"Warning: HD95 > 20 for image {image_names[i]}: {hd_:.4f}")
                if hd_ > 50:
                    if image_names:
                        hd_image_names.append(image_names[i])
                
                # calculate FOM
                pred_edge = mask2edge(pred)
                gt_edge = mask2edge(gt)
                fom_list.append(compute_fom(pred_edge, gt_edge))
        
        # output results
        print('|  Hd: {:.4f}'.format(np.mean(hd_list)).ljust(print_num_minus, ' '), '|')
        print('|  Sd: {:.4f}'.format(np.mean(sd_list)).ljust(print_num_minus, ' '), '|')
        # print('| FOM: {:.4f}'.format(np.mean(fom_list)).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        
        # output abnormal HD values and corresponding image names
        print(f"| abnormal HD95: {max_hd:.4f} (pic: {hd_image_names})")
        print('-' * print_num)

def eval_structure(pred_list, mask_list):
    """结构平滑性评估 (TV指标)[8](@ref)"""
    tv_pred, tv_ratio = [], []
    
    for pred, mask in zip(pred_list, mask_list):
        # 转换为二值图像 (确保0-1范围)
        pred_bin = pred.astype(bool)
        mask_bin = mask.astype(bool)
        
        # TV计算
        tv_p = compute_tv(pred_bin)
        # tv_p = compute_tv(pred_bin) / pred_bin.size
        tv_m = compute_tv(mask_bin)
        # tv_pred.append(tv_p)
        tv_ratio.append(tv_p / (tv_m + 1e-6))  # 防除零
    
    # 结果输出
    print('\n[结构评估]')
    # print('| TV_pred: {:.1f}±{:.1f}'.format(np.mean(tv_pred), np.std(tv_pred)))
    print('| TV_ratio: {:.3f}±{:.3f}'.format(np.mean(tv_ratio), np.std(tv_ratio)))

def eval_pixel(mask_list, seg_result_list, num_classes):
    c = confusion_matrix(mask_list, seg_result_list)

    hist_diag = np.diag(c)
    hist_sum_0 = c.sum(axis=0)
    hist_sum_1 = c.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    print('-' * print_num)
    if num_classes > 2:
        m_jaccard = np.nanmean(jaccard)
        m_dice = np.nanmean(dice)
        np.set_printoptions(precision=4, suppress=True)
        print('|  Jc: {}'.format(jaccard).ljust(print_num_minus, ' '), '|')
        print('|  Dc: {}'.format(dice).ljust(print_num_minus, ' '), '|')
        print('| mJc: {:.4f}'.format(m_jaccard).ljust(print_num_minus, ' '), '|')
        print('| mDc: {:.4f}'.format(m_dice).ljust(print_num_minus, ' '), '|')
    else:
        print('| Jc: {:.4f}'.format(jaccard[1]).ljust(print_num_minus, ' '), '|')
        print('| Dc: {:.4f}'.format(dice[1]).ljust(print_num_minus, ' '), '|')

def compute_ssim_mask(pred, gt, win_size=11):
    """针对二值掩码的SSIM计算（优化版）"""
    # 转换为0-1浮点类型
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    
    # 滑动窗口参数设置
    gaussian_weights = True
    data_range = 1.0  # 二值图像范围
    
    # 边界处理避免全0/全1情况
    if np.all(pred == pred[0,0]) or np.all(gt == gt[0,0]):
        return 1.0 if np.array_equal(pred, gt) else 0.0
    
    return ssim(pred, gt, win_size=win_size, 
                data_range=data_range, 
                gaussian_weights=gaussian_weights)

def eval_ssim(pred_list, mask_list):
    """SSIM评估模块"""
    ssim_values = []
    for pred, mask in zip(pred_list, mask_list):
        # # 统一尺寸（可选，根据实际需求）
        # if pred.shape != mask.shape:
        #     pred = A.resize(pred, mask.shape[0], mask.shape[1])
        
        # 单样本计算
        ssim_val = compute_ssim_mask(pred, mask)
        ssim_values.append(ssim_val)
    
    # 结果输出
    print('\n[结构相似性评估]')
    print(f'| SSIM: {np.mean(ssim_values):.4f}±{np.std(ssim_values):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='output_CTC_pre')
    parser.add_argument('--mask_path', default='cropped_mask_CTC')
    parser.add_argument('--if_3D', default=False)
    parser.add_argument('--resize_shape', default=None)
    parser.add_argument('--num_classes', default=2)
    args = parser.parse_args()

    pred_list = []
    mask_list = []
    image_names = []  # 存储图片名称

    pred_flatten_list = []
    mask_flatten_list = []

    num = 0

    for i in os.listdir(args.pred_path):
        pred_path = os.path.join(args.pred_path, i)
        mask_path = os.path.join(args.mask_path, i)
        
        if args.if_3D:
            pred = sitk.ReadImage(pred_path)
            pred = sitk.GetArrayFromImage(pred)
            #概率prediction阈值化
            pred = (pred > 0.89).astype(np.uint8)
            mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask)

        else:
            # pred = (sitk.GetArrayFromImage(sitk.ReadImage(pred_path)) > 1)
            # mask = (sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) > 1)
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask = (mask > 0.89).astype(np.uint8)  # 确保mask是二值化的
            pred = (pred*mask > 0.5).astype(np.uint8)
            
            if pred.shape != mask.shape:
                print("预测结果和真实标签的形状不匹配，需要调整大小")
                exit()

        pred_list.append(pred)
        mask_list.append(mask)
        image_names.append(i)  # 保存图片名称

        if num == 0:
            pred_flatten_list = pred.flatten()
            mask_flatten_list = mask.flatten()
        else:
            pred_flatten_list = np.append(pred_flatten_list, pred.flatten())
            mask_flatten_list = np.append(mask_flatten_list, mask.flatten())

        num += 1
    
    eval_pixel(mask_flatten_list, pred_flatten_list, args.num_classes)
    eval_distance(mask_list, pred_list, args.num_classes, image_names)  # 传递图片名称
    eval_structure(pred_list, mask_list)  # 新增结构评估
    # eval_ssim(pred_list, mask_list)  # 新增SSIM评估