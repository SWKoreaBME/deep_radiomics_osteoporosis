import numpy as np
import os
import pickle as pkl
import pandas as pd
import torch

from glob import glob
from tqdm import tqdm


def read_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        return_item = pkl.load(f)
    return return_item


def save_pickle(pickle_object, pickle_file):
    with open(pickle_file, 'wb') as f:
        pkl.dump(pickle_object, f)


def read_csv(csv_file):
    return pd.read_csv(csv_file)


def csv_preprocess(df):
    return df.fillna(df.mean())


def dict_to_array(dict):
    pbar = tqdm(len(dict))
    for dict_idx, (key, value) in enumerate(dict.items()):
        if value.shape[0] != 920:
            continue
        if dict_idx == 0:
            return_values = np.expand_dims(value, 0)
        else:
            return_values = np.append(return_values, np.expand_dims(value, 0), 0)
        pbar.update(1)
    return return_values


def get_subject_names(df, phase):
    if phase != "brmh":
        subjects = np.unique(np.array(['_'.join(x.split('_')[:2]) for x in df["Unnamed: 0"]]))
    else:
        subjects = np.unique(np.array(['_'.join(x.split('_')[:1]) for x in df["Unnamed: 0"]]))
    return subjects


def get_column_names(df, banned_cols=['Unnamed: 0', 'label']):
    columns = [x for x in df.columns if x not in banned_cols]
    return columns


def osteodataloader(X, Y, batch_size):
    dataloader = []
    batch_num = (X.shape[0] // batch_size) + 1

    for i in range(batch_num):

        batch_index = i * batch_size
        try:
            if Y is not None:
                single_batch_x, single_batch_y = X[batch_index:batch_index + batch_size], Y[
                                                                                          batch_index:batch_index + batch_size]
            else:
                single_batch_x = X[batch_index:batch_index + batch_size]
        except:
            if Y is not None:
                single_batch_x, single_batch_y = X[batch_index:], Y[batch_index:]
            else:
                single_batch_x = X[batch_index:]

        if Y is not None:
            dataloader.append([torch.tensor(single_batch_x).float(), torch.tensor(single_batch_y)])
        else:
            dataloader.append(torch.tensor(single_batch_x).float())

    return dataloader


if __name__ == '__main__':
    pass
