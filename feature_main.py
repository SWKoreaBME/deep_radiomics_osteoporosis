import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pickle as pkl
import cv2
import argparse
from tqdm import tqdm
import imageio

from utils.regions import getCorticalArea, getNeckArea, halfcut, cut_by_bbox, is_empty_image, crop_bottom
from utils.feature_extraction_utils import feature_extract, apply_wavelet, zscore_normalize
from utils.feature_utils import mask2binary

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Feature Extraction Main code')
    parser.add_argument('--image_dir', help='Image directory consists of .png files', required=True)
    parser.add_argument('--mask_dir', help='Mask directory consists of .png files', required=True)
    parser.add_argument('--label_file', help='label file path which has labels of images')
    parser.add_argument('--resize_factor', help='resizing factor (i.e. 70% -> 0.7)', default=1, type=float)
    parser.add_argument('--image_save_dir', help='Path to save images')
    parser.add_argument('--ext', help='external validation dataset', action='store_true', default=False)
    parser.add_argument('--brmh', help='institutional external validation dataset', action='store_true', default=False)
    parser.add_argument('--save_image', help='Save image to file', action='store_true', default=False)
    parser.add_argument('--save_file', help='save file name', required=True)

    args = parser.parse_args()
    # 잘리지 않은 영상과 ( 원본 사이즈 ) 마스크가 input 으로 들어간다
    image_list = sorted(glob(os.path.join(args.image_dir, '*.png')))
    mask_list = sorted(glob(os.path.join(args.mask_dir, '*.png')))

    # error subjects for a recently configured dataset
    error_subjects = ['2012_0078', '2014_0706', '2017_0658', '2019_1203', '2019_1238']

    image_list = [x for x in image_list if x.split('/')[-1].rstrip('.png') not in error_subjects]
    mask_list = [x for x in mask_list if x.split('/')[-1].rstrip('_mask.png') not in error_subjects]

    if not (args.ext or args.brmh):
        with open('/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle', 'rb') as f:
            labels = pkl.load(f)

    if len(image_list) != len(mask_list):
        mask_subjects = [x.split('/')[-1].rstrip('_mask.png') for x in mask_list]
        image_list = [x for x in image_list if x.split('/')[-1].rstrip('.png') in mask_subjects]

    image_list = sorted(image_list)
    mask_list = sorted(mask_list)

    length_of_roi = 1000
    whole_original = []
    whole_neck = []
    whole_cortical = []

    subjects = []

    factor = args.resize_factor
    length_of_roi = int(length_of_roi * factor)
    errors = []
    print(f"\n {len(image_list)} number of image will be processed --- \n")

    with tqdm(total=len(image_list)) as pbar:

        for index, (image, mask) in enumerate(zip(image_list, mask_list)):
            pbar.update(1)
            file_name = image.split('/')[-1]
            subject_name = file_name.rstrip('.png')

            try:
                if not (args.ext or args.brmh):
                    try:
                        label = labels[subject_name]
                    except:
                        errors.append(subject_name)
                        continue

                if mask.split('/')[-1].rstrip('_mask.png') != subject_name:
                    print(mask, subject_name)
                    print(f"{subject_name} error !!")
                    errors.append(subject_name)
                    continue

                img = cv2.imread(image)[:, :, 0]
                mask = cv2.imread(mask)[:, :, 0]

                # Resizing
                mask = cv2.resize(mask, (int(img.shape[1] * factor), int(img.shape[0] * factor)))
                img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))

                mask = mask2binary(mask)
                roi = zscore_normalize(img, mask)

                for half_index, (mask_arr, img_arr) in enumerate(zip(halfcut(mask), halfcut(roi))):
                    
                    half_original = []
                    file_name_half = file_name.replace('.png', '_' + str(half_index))

                    if half_index == 0:
                        side = 'left'
                    elif half_index == 1:
                        side = 'right'

                    if is_empty_image(mask_arr, 0.05):
                        continue
                        # if not img_arr.shape == mask_arr.shape: continue

                    mask_arr = np.where(mask_arr > np.max(mask_arr) * 0.8, 1, 0)
                    mask_arr, img_arr = cut_by_bbox(input_mask=mask_arr, input_original=img_arr)

                    # crop bottom
                    mask_arr = crop_bottom(mask_arr, ratio=1.5, avg_h = 1334 * factor)
                    img_arr = crop_bottom(img_arr, ratio=1.5, avg_h = 1334 * factor)
                    
                    roi_arr = np.multiply(img_arr, mask_arr) if img_arr.shape == mask_arr.shape else img_arr

                    top_y_pixel = np.min(np.where(mask_arr > 0)[0])

                    roi_arr = roi_arr[top_y_pixel:top_y_pixel + length_of_roi, :]
                    mask_arr = mask_arr[top_y_pixel:top_y_pixel + length_of_roi, :]
                    original_features, original_columns = feature_extract(roi_arr, mask_arr)

                    subjects.append(subject_name + '_' + str(half_index))

                    if not (args.ext or args.brmh):
                        half_original.extend([label] + original_features)

                    else:
                        half_original.extend(original_features)

                    for index, wv in enumerate(apply_wavelet(roi_arr)):
                        wv = cv2.resize(wv, (mask_arr.shape[1], mask_arr.shape[0]), interpolation=cv2.INTER_CUBIC)
                        original_features, wv_cols = feature_extract(wv, mask_arr, ['firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm'])
                        half_original.extend(original_features)
                        original_columns.extend([x + '_wv_' + str(index) for x in wv_cols])

                    whole_original.append(half_original)

                    if not (args.ext or args.brmh):
                        columns = ['label'] + original_columns

                    else:
                        columns = original_columns

                # Save as csv file
                    if not args.save_image:
                        pd.DataFrame(np.array(whole_original), columns=columns, index=subjects).to_csv(
                            f'/sdb1/share/ai_osteoporosis_hip/real_final_png/texture_feature/original_{args.save_file}.csv')

            except Exception as e:
                print(e)
                errors.append(subject_name)
                np.save('./log/errors_{}.npy'.format(args.save_file), np.array(errors))
                continue