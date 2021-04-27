from skimage.measure import label, regionprops
import math
import nibabel as nib
import numpy as np
import radiomics
import logging

radiomics.logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

def cut_by_bbox(input_mask, input_original):
    '''
        cut input image by skimage.regionprops.bbox

        - input_image ( numpy array ) : must be binary
    '''

    input_mask = input_mask[:, :, 0] if len(input_mask.shape) == 3 else input_mask
    input_original = input_original[:, :, 0] if len(input_original.shape) == 3 else input_original

    if len(np.unique(input_mask, return_index=False)) != 2: return False

    minr, minc, maxr, maxc = regionprops(input_mask)[0].bbox
    return input_mask[minr:maxr, minc:maxc], input_original[minr:maxr, minc:maxc]

def degree(x):
    return math.degrees(x)


def getNeckArea(mask_arr, img_arr, width, height, side):
    '''
        It shows the major and minor axis of input mask especially femur head

        - mask_path : binary image path ( mask, .jpg .png ,,, any type which can be read by cv2 )
    '''

    sample_mask = mask_arr[:, :, 0] if len(mask_arr.shape) == 3 else mask_arr
    sample_image = img_arr[:, :, 0] if len(img_arr.shape) == 3 else img_arr

    if len(np.unique(sample_mask)) != 2:
        binary_mask = np.where(sample_mask > 200, 255, 0)
    else:
        binary_mask = sample_mask

    mask, image_original = cut_by_bbox(input_mask=binary_mask, input_original=sample_image)

    regions = regionprops(mask)

    props = regions[0]
    y0, x0 = props.centroid
    orientation = props.orientation

    if side == 'left':
        x1 = x0 + math.cos(orientation) * 0.15 * props.major_axis_length
        y1 = y0 + math.sin(orientation) * 0.7 * props.major_axis_length
    elif side == 'right':
        x1 = x0 - math.cos(orientation) * 0.15 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.7 * props.major_axis_length
        
    filled = np.zeros_like(mask_arr)
    filled[int(y1 - height * 0.5):int(y1 + height * 0.5), int(x1 - width * 0.5):int(x1 + width * 0.5)] = 1

    return filled

def RotateNib(file):
    return np.rot90(np.flip(nib.load(file).get_fdata(), 1), 1)

def halfcut(arr):
    arr = arr[:, :, 0] if len(arr.shape) != 2 else arr
    h, w = arr.shape
    half_left = arr[:, :int(w / 2)]
    half_right = arr[:, int(w / 2):]

    return half_left, half_right

def is_empty_image(arr, threshold=0.1):
    (x, y) = np.nonzero(arr)
    return len(x) < (arr.shape[0] * arr.shape[1] * threshold)

def readManualText(img_txt_path):
    coords = []
    with open(img_txt_path, 'rb') as f:
        for index, line in enumerate(f):
            if index >= 1:
                line = str(line).lstrip("b'").rstrip("'").rstrip("\\r\\n")
                coords.append(line.split(","))
    return np.array(coords, np.float)

def getManualBbox(img_arr, img_txt_path):
    coords = readManualText(img_txt_path)
    mask = np.zeros_like(img_arr)
    for coord in coords:
        top_left = coord[:2].astype(np.int)
        width, height = coord[2:].astype(np.int)

        mask[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width] = 1

    return mask

def getThresholdImage(roi_arr, tr):
    return np.where(roi_arr > tr, roi_arr, 0)

def getCorticalBbox(mask_arr, width, height):
    filled = np.zeros_like(mask_arr)

    coords = np.array(np.where(mask_arr != 0))
    left_y = np.max(coords[0])

    bottom_left = [x for x in coords.T if x[0] == left_y][0]
    bottom_right = [bottom_left[0], bottom_left[1] + width]

    top_left = [bottom_left[0] - height, bottom_left[1]]
    top_right = [bottom_left[0] - height, bottom_left[1] + width]

    filled[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1
    return filled

def getCorticalArea(mask_arr, roi_arr, width, height):
    cortical_arr = getCorticalBbox(mask_arr, width=width, height=height)
    return cortical_arr

def crop_bottom(arr, ratio=1.5, avg_h = 1334):
    is_short = arr.shape[0] <= avg_h
    
    if is_short:
        return arr
    else:
        return arr[:int(arr.shape[1] * ratio), :]