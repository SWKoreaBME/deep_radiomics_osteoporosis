import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

def dummy_labelize_swk(data, n_classes):

    """
        Make labels into dummy form

        (example)

        input : [0, 1, 2, 0, 0]
        output : [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]
    
    Returns:
        [array] -- [dummy for of labels]
    """
    
    label = np.zeros((len(data), n_classes), dtype=int)
    
    for k, i in zip(data, label):
        i[k] = 1
    
    return label

def boxplot():

    # all results box plot

    sns.set(style='whitegrid')

    plt.figure(figsize=(70,50))

    b = sns.boxplot(data=whole_rocs)
    b.set_xlabel("number of k",fontsize=40)
    b.set_ylabel("auc score",fontsize=40)
    b.tick_params(labelsize=30)

    return

def mask_oversize(sample_image):
    """half image --> fill rest of the half image with zero : with Full size 
    
    Arguments:
        sample_image {[type]} -- [description]
    
    Returns:
        result_image : full image with numpy array
    """
    img_arr = imageio.imread(sample_image)
    null_image = np.multiply(img_arr, 0)
    
    if 'left' in sample_image :
        result_image = np.hstack([null_image, img_arr])
    elif 'right' in sample_image :
        result_image = np.hstack([img_arr, null_image])
        
    return result_image

def mask2binary(arr, tr=0.7):
    arr = np.where(arr<=np.max(arr)*tr, 0, arr)
    arr = np.where(arr>np.max(arr)*tr, 1, arr)
    return arr.astype(np.int)