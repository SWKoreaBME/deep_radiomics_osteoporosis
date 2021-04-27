import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
import os
import argparse

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import cv2

from random import shuffle, sample


class TensorBoard():
    def __init__(self, log_dir):
        super(TensorBoard, self).__init__()
        self.tb = SummaryWriter(log_dir=log_dir)

    def update(self, field, value, epoch):
        self.tb.add_scalar(field, value, epoch)

def ConvBlock3(inp, out, stride, pad):
    """
    3x3 ConvNet building block with different activations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(True)
    )


def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 2, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block2 = nn.Sequential(ConvBlock3(bw, bw * 2, 1, 0),
                                    ConvBlock3(bw * 2, bw * 2, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block3 = ConvBlock3(bw * 2, bw * 4, 1, 0)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return F.avg_pool2d(o3, o3.size()[2:]).view(x.size(0), -1)


class OsteoSideModel(nn.Module):
    def __init__(self,
                 bw,
                 drop=0.3,
                 num_classes=1,
                 use_w_init=True,
                 use_clinic=False,
                 return_fc=False):

        super().__init__()
        self.branch = Branch(bw)
        self.use_clinic = use_clinic
        self.return_fc = return_fc
        if self.use_clinic:
            print("Use clinic data !")

        if num_classes == 1:
            softmaxer = nn.Sigmoid()
        elif num_classes == 2:
            softmaxer = nn.Softmax()

        fc_in = bw * 4 + 4 if self.use_clinic else bw * 4

        if drop > 0:
            self.final = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(fc_in, num_classes),
                softmaxer
            )

        else:
            self.final = nn.Sequential(
                nn.Linear(fc_in, num_classes),
                softmaxer
            )

        # Custom weights initialization
        if use_w_init:
            self.apply(weights_init_uniform)

    def forward(self, x1, clinic=None, save=False):
        # Shared weights
        o1 = self.branch(x1)  # left

        if save:
            self.left_value = o1

        feats = torch.cat([o1, clinic], 1) if self.use_clinic else o1

        if self.return_fc:
            return self.final(feats), feats
        else:
            return self.final(feats)


class OsteoSideDataset(torch.utils.data.Dataset):
    """Some Information about OsteoSiameseDataset"""
    def __init__(self,
                 data_list,
                 label_file="/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle",
                 clinic_file="/home/jarvis/Desktop/swk/Data/clinic_data/whole_clinic_final.pickle"):

        super(OsteoSideDataset, self).__init__()

        self.data_list = data_list
        # self.data_list = glob(os.path.join(data_dir, '*.png'))

        with open(label_file, 'rb') as f:
            self.label_dict = pkl.load(f)

        with open(clinic_file, 'rb') as f:
            self.clinic_dict = pkl.load(f)

    def read_image(self, img_file, target_size=[448, 224]):
        """
            input : img file ( path )
            output : Transformed img array
        """

        img = np.fliplr(cv2.imread(img_file)[:, :, 0]) if img_file.endswith('_1.png') else cv2.imread(img_file)[:, :, 0]
        img = cv2.resize(img, (target_size[1], target_size[0]))

        # Min-Max Normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = np.broadcast_to(img, (1, *target_size))

        # todo: If Longer one, cut lower part of the image
        # todo: We already use cropped image, don't have to crop more.
        # img = img[:int(img.shape[0] * 0.9), :].astype(np.float32)

        return torch.Tensor(img.copy()).type(torch.FloatTensor)

    def __getitem__(self, index):

        img_name = self.data_list[index]

        # subject_name
        subject_name = '_'.join(img_name.split('/')[-1].split('_')[:2])
        try:
            clinic_data = self.clinic_dict[subject_name]
        except:
            clinic_data = self.clinic_dict[subject_name.split('_')[0]]

        try:
            label = self.label_dict[subject_name]
        except:
            label = 0

        try:
            data = dict(
                image=self.read_image(img_name),
                label=label,
                clinic=clinic_data,
                subject_name=subject_name,
                img_name=img_name
            )

        except Exception as e:
            data = dict(
                image=self.read_image(img_name),
                label=label,
                clinic=clinic_data,
                subject_name=subject_name,
                img_name=img_name
            )
            print(img_name, "got error {}".format(e))
            pass

        return data

    def __len__(self):
        return len(self.data_list)


def logger(acc_score,
           epoch_idx,
           batch_idx,
           num_epoch,
           num_batches,
           total_loss,
           auc_score,
           phase='train',
           tb=None):

    log_fmt = "--{}--\n"\
              + "Epoch [{}/{}] Batch [{}/{}]\n"\
              + "loss -- {:.4f} Acc -- {:.4f} AUC -- {:.4f}"

    print(log_fmt.format(phase,
                         epoch_idx+1,
                         num_epoch,
                         batch_idx,
                         num_batches,
                         total_loss.data,
                         acc_score,
                         auc_score))

    if tb is not None:
        tb.update('Accuracy/{}'.format(phase), acc_score, epoch_idx+1)
        tb.update('Loss/{}'.format(phase), total_loss.data, epoch_idx+1)
        tb.update('AUC/{}'.format(phase), auc_score, epoch_idx+1)


def split_train_test(data_dir,
                     balance=True,
                     train_ratio=0.8,
                     ros=True,
                     label_file="/sdb1/share/ai_osteoporosis_hip/real_final_png/label_dict_final.pickle"):

    error_list = [
        '2012_0078',
        '2014_0706',
        '2017_0658',
        '2019_1203',
        '2019_1238'
    ]

    get_subject_name = lambda x: '_'.join(x.split('/')[-1].split('_')[:2])

    subject_list = list(set([get_subject_name(x) for x in glob(os.path.join(data_dir, '*.png')) if get_subject_name(x)
                             not in error_list]))

    print("A total number of subjects :", len(subject_list))

    if balance:
        with open(label_file, 'rb') as f:
            label_dict = pkl.load(f)

        pos_list = [x for x in subject_list if label_dict[x] == 1]
        neg_list = [x for x in subject_list if label_dict[x] == 0]

        shuffle(pos_list)
        shuffle(neg_list)

        # todo: Split positive list, negative list
        pos_train, pos_valid = pos_list[:int(len(pos_list) * train_ratio)], pos_list[int(len(pos_list) * train_ratio):]
        neg_train, neg_valid = neg_list[:int(len(neg_list) * train_ratio)], neg_list[int(len(neg_list) * train_ratio):]

        if ros:

            pos_extended = []

            for i in range(len(neg_train) // len(pos_train)):
                pos_extended.extend(sample(pos_train, len(pos_train)))

            train_list = neg_train + pos_extended

        else:
            train_list = pos_train + neg_train

        valid_list = pos_valid + neg_valid

    else:
        raise NotImplementedError

    train_image_list = []
    valid_image_list = []

    for train_subject in train_list:
        left_image = os.path.join(data_dir, train_subject+"_0.png")
        right_image = os.path.join(data_dir, train_subject+"_1.png")

        if os.path.exists(left_image) and os.path.exists(right_image):
            train_image_list.append(left_image)
            train_image_list.append(right_image)

    for valid_subject in valid_list:
        left_image = os.path.join(data_dir, valid_subject+"_0.png")
        right_image = os.path.join(data_dir, valid_subject+"_1.png")

        if os.path.exists(left_image) and os.path.exists(right_image):
            valid_image_list.append(left_image)
            valid_image_list.append(right_image)

    print("Train images: {} (Pos:{}, Neg:{}), Validation images: {} (Pos:{}, Neg:{})".format(len(train_image_list),
                                                                                             len(pos_extended),
                                                                                             len(neg_train),
                                                                                             len(valid_image_list),
                                                                                             len(pos_valid),
                                                                                             len(neg_valid)))

    print("Baseline Accuracy : {:.4f}".format(len(neg_valid) / (len(pos_valid) + len(neg_valid))))

    return train_image_list, valid_image_list


def auc(output, target):
    with torch.no_grad():
        return roc_auc_score(target.cpu().numpy(), output.cpu().numpy(), multi_class='ovo', average='macro')


def train(model,
          device,
          num_epoch,
          batch_size,
          num_workers,
          data_dir='/sdb1/share/ai_osteoporosis_hip/real_final_png/dl_cropped/development/',
          save='./cp',
          train_ratio=0.8,
          ros=True):

    train_list, val_list = split_train_test(data_dir,
                                            balance=True,
                                            train_ratio=train_ratio,
                                            ros=ros)

    best_result: float = 0

    tb = TensorBoard(log_dir=args.log_dir)

    train_dataset = OsteoSideDataset(data_list=train_list)
    val_dataset = OsteoSideDataset(data_list=val_list)

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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), betas=[0.9, 0.999], eps=1e-5)

    # todo: Train Code
    epoch_pbar = tqdm(total=num_epoch)
    for epoch_idx in range(num_epoch):
        epoch_pbar.update(1)

        total_epoch_loss = 0
        total_num_correct = 0
        total_num_image = 0

        train_pbar = tqdm(total=len(train_loader))
        model.train()
        for idx, data in enumerate(train_loader, start=1):

            train_pbar.update(1)

            image = data['image'].to(device)
            # clinic = data['clinic'].to(device)
            label = data['label'].to(device)
            output = model(image)

            loss = criterion(torch.squeeze(output), label.float())
            loss.backward()
            optimizer.step()

            # Give threshold to output
            preds = torch.squeeze((output > 0.5).int())
            total_num_correct += (preds == label).sum().item()
            total_num_image += image.size(0)
            total_epoch_loss += loss.data * image.size(0)

            auc_score = auc(output, label)
            print("Train | Num Correct: {}, Total Num Image: {}".format(total_num_correct, total_num_image))
            acc_score = total_num_correct / total_num_image

            if idx % 10 == 0:
                logger(acc_score=acc_score,
                       epoch_idx=epoch_idx,
                       batch_idx=idx,
                       num_epoch=num_epoch,
                       num_batches=len(train_loader),
                       total_loss=total_epoch_loss/total_num_image,
                       auc_score=auc_score,
                       tb=tb)

        train_pbar.close()

        # Validation
        model.eval()
        with torch.no_grad():

            total_epoch_loss = 0
            total_num_correct = 0
            total_num_image = 0

            valid_pbar = tqdm(total=len(val_loader))

            outputs = []
            labels = []

            for idx, data in enumerate(val_loader, start=1):

                valid_pbar.update(1)

                image = data['image'].to(device)
                # clinic = data['clinic'].to(device)
                label = data['label'].to(device)
                output = model(image)

                loss = criterion(torch.squeeze(output), label.float())

                # Give threshold to output

                preds = torch.squeeze((output > 0.5).int())
                total_num_correct += (preds == label).sum().item()
                total_num_image += image.size(0)
                total_epoch_loss += loss.data * image.size(0)

                labels.extend(list(label.cpu().numpy()))
                outputs.extend(list(output.cpu().numpy()))

                print("Valid | Num Correct: {}, Total Num Image: {}".format(total_num_correct, total_num_image))

            auc_score = auc(torch.Tensor(np.array(outputs)),
                            torch.Tensor(np.array(labels)))

            acc_score = total_num_correct / total_num_image

            logger(acc_score=acc_score,
                   epoch_idx=epoch_idx,
                   batch_idx=idx,
                   num_epoch=num_epoch,
                   num_batches=len(val_loader),
                   total_loss=total_epoch_loss/total_num_image,
                   auc_score=auc_score,
                   phase='Valid',
                   tb=tb)

            val_acc = total_num_correct / total_num_image
            valid_pbar.close()

            if val_acc > best_result:
                best_result = val_acc
                torch.save(model, os.path.join(save, 'best.pth'))

            # save model
            torch.save(model, os.path.join(save, 'checkpoint.pth'))

    epoch_pbar.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Whole Image Training")

    parser.add_argument('--num_epoch', '-e', dest='num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', '-w', dest='num_workers', type=int, default=8)
    parser.add_argument('--log_dir', '-d', dest='log_dir', type=str, default='./runs/')
    parser.add_argument('--pretrained', '-p', dest='pretrained', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device :", device)

    # todo: Model Selector
    model = OsteoSideModel(32)
    print("Model Ready to be trained")

    # train
    train(model=model,
          device=device,
          batch_size=args.batch_size,
          num_epoch=args.num_epoch,
          num_workers=args.num_workers)
