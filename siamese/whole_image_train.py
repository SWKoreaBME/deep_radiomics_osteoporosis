import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from glob import glob
import os
import argparse

from torch.utils.tensorboard import SummaryWriter

from whole_model import BinaryClassifier

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import cv2


class TensorBoard():
    def __init__(self, log_dir):
        super(TensorBoard, self).__init__()
        self.tb = SummaryWriter(log_dir=log_dir)

    def update(self, field, value, epoch):
        self.tb.add_scalar(field, value, epoch)


class OsteoWholeDataset(torch.utils.data.Dataset):
    """Some Information about OsteoSiameseDataset"""
    def __init__(self,
                 data_list,
                 crop=False,
                 label_file="/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle",
                 clinic_file="/home/jarvis/Desktop/swk/Data/clinic_data/whole_clinic_final.pickle"):

        super(OsteoWholeDataset, self).__init__()

        self.crop = crop
        self.data_list = data_list
        # self.data_list = glob(os.path.join(data_dir, '*.png'))

        with open(label_file, 'rb') as f:
            self.label_dict = pkl.load(f)

        with open(clinic_file, 'rb') as f:
            self.clinic_dict = pkl.load(f)

    def read_image(self, img_file, target_size=(512, 512)):
        """
            input : img file ( path )
            output : Transformed img array
        """

        img = cv2.imread(img_file)[:, :, 0]
        img = cv2.resize(img, target_size)

        # Min-Max Normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = np.broadcast_to(img, (3, 512, 512))

        # todo: If Longer one, cut lower part of the image
        # img = img[:int(img.shape[0] * 0.9), :].astype(np.float32)

        return torch.Tensor(img.copy()).type(torch.FloatTensor)

    def __getitem__(self, index):

        img_name = self.data_list[index]

        # subject_name
        subject_name = img_name.split('/')[-1].replace('.png', '')
        clinic_data = self.clinic_dict[subject_name]
        label = self.label_dict[subject_name]

        try:
            data = dict(
                image=self.read_image(img_name),
                label=label,
                clinic=clinic_data,
                subject_name=subject_name
            )

        except Exception as e:
            data = dict(
                image=self.read_image(img_name),
                label=label,
                clinic=clinic_data,
                subject_name=subject_name
            )
            print(img_name, "got error {}".format(e))
            pass

        return data

    def __len__(self):
        return len(self.data_list)


def logger(num_correct,
           num_image,
           epoch_idx,
           batch_idx,
           num_epoch,
           num_batches,
           total_loss,
           phase='train',
           fold_idx=None,
           tb=None):

    log_fmt = "--{}--\n"\
              + "Epoch [{}/{}] Batch [{}/{}]\n"\
              + "loss -- {:.4f} Acc -- {:.4f}"

    print(log_fmt.format(phase,
                         epoch_idx,
                         num_epoch,
                         batch_idx,
                         num_batches,
                         total_loss.data/num_image,
                         num_correct/num_image))

    if tb is not None:
        tb.update('Accuracy/{}/cv_{}'.format(phase, fold_idx+1), num_correct/num_image, epoch_idx)
        tb.update('Loss/{}/cv_{}'.format(phase, fold_idx+1), total_loss.data/num_image, epoch_idx)


def split_train_test(data_dir,
                     num_fold: int = 5,
                     balance=True,
                     label_file="/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle"):

    error_list = [
        '2012_0078',
        '2014_0706',
        '2017_0658',
        '2019_1203',
        '2019_1238'
    ]

    whole_list = [x for x in glob(os.path.join(data_dir, '*.png')) if
                  x.split('/')[-1].replace('.png', '') not in error_list]

    split_list = dict()

    if balance:
        with open(label_file, 'rb') as f:
            label_dict = pkl.load(f)

        pos_list = [x for x in whole_list if label_dict[x.split('/')[-1].replace('.png', '')] == 1]
        neg_list = [x for x in whole_list if label_dict[x.split('/')[-1].replace('.png', '')] == 0]

        pos_num_per_fold = len(pos_list) // num_fold
        neg_num_per_fold = len(neg_list) // num_fold

        for i in range(num_fold):
            if i == num_fold-1:
                split_list[i + 1] = pos_list[i * pos_num_per_fold:] + neg_list[i * neg_num_per_fold:]
            else:
                split_list[i + 1] = pos_list[i * pos_num_per_fold:(i + 1) * pos_num_per_fold] \
                                    + neg_list[i * neg_num_per_fold:(i + 1) * neg_num_per_fold]
        return split_list

    else:
        num_per_fold = len(whole_list) // num_fold
        for i in range(num_fold):
            if i == num_fold - 1:
                split_list[i + 1] = whole_list[i * num_per_fold:]
            else:
                split_list[i + 1] = whole_list[i * num_per_fold:(i + 1) * num_per_fold]
        return split_list


def train(model,
          device,
          num_epoch,
          batch_size,
          num_workers,
          data_dir='/sdb1/share/ai_osteoporosis_hip/real_final_png/dl_cropped/',
          kfold=1,
          save='./cp'):

    whole_list = split_train_test(data_dir,
                                  num_fold=kfold,
                                  balance=True)

    valid_result = dict()
    best_result: float = 0

    tb = TensorBoard(log_dir=args.log_dir)

    for fold_idx in range(kfold):
        print("CV Training -- [{}/{}]".format(fold_idx+1, kfold))

        train_list, val_list = [], []
        for key, value in whole_list.items():
            if key == fold_idx+1:
                val_list = value
            else:
                train_list.extend(value)

        train_dataset = OsteoWholeDataset(data_list=train_list)
        val_dataset = OsteoWholeDataset(data_list=val_list)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)

        # todo: Train Setting

        model = nn.DataParallel(model.to(device))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), betas=[0.9, 0.999], eps=1e-5)

        # todo: Train Code
        for epoch_idx in range(num_epoch):

            total_epoch_loss = 0
            total_num_correct = 0
            total_num_image = 0

            for idx, data in enumerate(train_loader, start=1):

                image = data['image'].to(device)
                # clinic = data['clinic'].to(device)
                label = data['label'].to(device)
                output = model(image)

                loss = criterion(torch.squeeze(output), label.float())
                loss.backward()
                optimizer.step()

                # Give threshold to output
                preds = (nn.Sigmoid()(output) > 0.5).float()
                total_num_correct += (preds == label).sum().item()
                total_num_image += image.size(0)
                total_epoch_loss += loss.data

                logger(num_correct=total_num_correct,
                       num_image=total_num_image,
                       epoch_idx=epoch_idx,
                       batch_idx=idx,
                       num_epoch=num_epoch,
                       num_batches=len(train_loader),
                       total_loss=total_epoch_loss,
                       fold_idx=fold_idx,
                       tb=tb)

        # Validation
        model.eval()
        with torch.no_grad():

            total_epoch_loss = 0
            total_num_correct = 0
            total_num_image = 0

            for idx, data in enumerate(val_loader, start=1):

                image = data['image'].to(device)
                # clinic = data['clinic'].to(device)
                label = data['label'].to(device)
                output = model(image)

                loss = criterion(torch.squeeze(output), label.float())

                # Give threshold to output

                preds = (nn.Sigmoid()(output) > 0.5).float()
                total_num_correct += (preds == label).sum().item()
                total_num_image += image.size(0)
                total_epoch_loss += loss.data

                logger(num_correct=total_num_correct,
                       num_image=total_num_image,
                       epoch_idx=1,
                       batch_idx=idx,
                       num_epoch=1,
                       num_batches=len(val_loader),
                       total_loss=total_epoch_loss,
                       phase='Valid',
                       fold_idx=fold_idx,
                       tb=tb)

            val_acc = total_num_correct / total_num_image

            valid_result[fold_idx] = val_acc
            if val_acc > best_result:
                best_result = val_acc
                torch.save(model, os.path.join(save, 'cv_best_{}.pth'.format(fold_idx+1)))

            # save model
            torch.save(model, os.path.join(save, 'cv_{}.pth'.format(fold_idx+1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Whole Image Training")

    parser.add_argument('--num_epoch', '-e', dest='num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', '-w', dest='num_workers', type=int, default=8)
    parser.add_argument('--num_fold', '-f', dest='num_fold', type=int, default=5)
    parser.add_argument('--log_dir', '-d', dest='log_dir', type=str, default='./runs/')
    parser.add_argument('--pretrained', '-p', dest='pretrained', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device :", device)

    # todo: Model Selector
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=args.pretrained)
    print("pretrained : {}".format(args.pretrained))
    model.classifier = BinaryClassifier()

    # train
    train(model=model,
          device=device,
          batch_size=args.batch_size,
          num_epoch=args.num_epoch,
          num_workers=args.num_workers,
          kfold=args.num_fold)

    # for img in tqdm(glob(os.path.join(data_dir, '*.png'))):
    #     try:
    #         cv2.imread(img)[:, :, 0]
    #         pass
    #     except Exception as e:
    #         print(e)
    #         errors.append(img)
    #         print(img)
    #         pass
    #
    # print(errors)
