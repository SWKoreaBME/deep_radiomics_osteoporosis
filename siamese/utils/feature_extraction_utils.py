import numpy as np
import SimpleITK as sitk
from radiomics import firstorder, glcm, shape, glrlm, glszm, ngtdm, gldm, featureextractor
import pywt
import radiomics

def zscore_normalize(image, mask):
    roi_arr = np.multiply(image, mask)
    x,y = np.where(mask != 0)
    mean, std = np.mean(roi_arr[x, y]), np.std(roi_arr[x, y])
    roi_arr = (roi_arr - mean) / std
    return roi_arr

def apply_wavelet(img_arr):
    coeffs2 = pywt.dwt2(img_arr, 'bior1.3')
    _, (LH, HL, HH) = coeffs2
    return LH, HL, HH

def read_image(img_arr, mask=True):
    if mask:
        img_arr = np.where(img_arr == np.max(img_arr), 1, 0)
    img_arr = np.expand_dims(img_arr, axis=2)
    im_3d = sitk.GetImageFromArray(img_arr)
    return im_3d

# Radiomic Feautures Extracting Functions

def feature_extract(image, mask, features=['firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm', 'shape']):
    '''
    :param image_origin: image_array (numpy array)
    :param image_mask: mask_array (numpy array)
    :subject: subject name
    :return: whole features, featureVector, make csv_file
    '''

    image = read_image(image, mask=False)
    mask = read_image(mask)

    settings = {}
    settings['binwidth'] = 25
    settings['resampledPixelSpacing'] = None
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True

    extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)
    extractor.settings['enableCExtensions'] = True

    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())

    featureVector = extractor.execute(image, mask)

    cols = [];
    feats = []
    for feature in features:
        for featureName in sorted(featureVector.keys()):
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])

    return feats, cols