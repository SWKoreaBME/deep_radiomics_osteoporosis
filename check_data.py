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

image_dir = '/sdb1/share/ai_osteoporosis_hip/real_final_png/development/'
mask_dir = '/sdb1/share/ai_osteoporosis_hip/real_final_png/mask/development_mask/'

image_list = sorted(glob(os.path.join(image_dir, '*.png')))
mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

# error subjects for a recently configured dataset
error_subjects = ['2012_0078', '2014_0706', '2017_0658', '2019_1203', '2019_1238']

image_list = [x for x in image_list if x.split('/')[-1].rstrip('.png') not in error_subjects]
mask_list = [x for x in mask_list if x.split('/')[-1].rstrip('_mask.png') not in error_subjects]

resize_factor = 0.25
ext = False
brmh = False

with tqdm(total=len(image_list)) as pbar:
    if not (ext or brmh):
        with open('/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle', 'rb') as f:
            labels = pkl.load(f)

        if len(image_list) != len(mask_list):
            mask_subjects = [x.split('/')[-1].rstrip('_mask.png') for x in mask_list]
            image_list = [x for x in image_list if x.split('/')[-1].rstrip('.png') in mask_subjects]

        length_of_roi = 1000
        whole_original = []
        whole_neck = []
        whole_cortical = []

        subjects = []

        factor = resize_factor
        length_of_roi = int(length_of_roi * factor)
        errors = []
        print(f"\n {len(image_list)} number of image will be processed --- \n")

        for image, mask in zip(image_list[400:], mask_list[:400]):
            pbar.update(1)
            file_name = image.split('/')[-1]
            subject_name = file_name.rstrip('.png')

            try:
                img = cv2.imread(image)[:, :, 0]
                mask = cv2.imread(mask)[:, :, 0]

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
                    print(roi_arr.shape, mask_arr.shape)

            except Exception as e:
                print(e)
                print(subject_name)
                errors.append(subject_name)
                continue

        print(errors)