"""
Network architecture class
Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
modified by SangWook Kim, Seoul National University Hospital, 2020.
"""

import os
import pickle as pkl
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18

from acm import ACM


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
        if m.bias is not None:
            m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResidualBranch(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.res18_block = resnet18(pretrained=pretrained)
        self.res18_block.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.res18_block.fc = Identity()

    def forward(self, x):
        return self.res18_block(x)


class Branch(nn.Module):
    def __init__(self, bw, use_acm=False):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 2, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block2 = nn.Sequential(ConvBlock3(bw, bw*2, 1, 0),
                                    ConvBlock3(bw*2, bw*2, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block3 = ConvBlock3(bw*2, bw*4, 1, 0)
        
        self.use_acm = use_acm
        
        if use_acm:
            self.acm1 = ACM(num_heads=bw//4, num_features=bw, orthogonal_loss=False)
            self.acm1.init_parameters()
            
            self.acm2 = ACM(num_heads=bw//2, num_features=bw*2, orthogonal_loss=False)
            self.acm2.init_parameters()

    def forward(self, x):
        o1 = self.acm1(self.block1(x)) if self.use_acm else self.block1(x)
        o2 = self.acm2(self.block2(o1)) if self.use_acm else self.block2(o1)
        o3 = self.block3(o2)
        return F.avg_pool2d(o3, o3.size()[2:]).view(x.size(0), -1)


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


class OsteoSiameseNet(nn.Module):
    """
    Siamese Net to automatically grade osteoarthritis
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    def __init__(self,
                 bw,
                 drop,
                 num_classes=1,
                 use_resnet=True,
                 pretrained=False,
                 use_w_init=True,
                 use_clinic=False,
                 use_acm=False,
                 return_fc=False):

        super().__init__()

        # Basic Block
        if use_resnet:
            self.branch = ResidualBranch(pretrained=pretrained)
            bw = 128
        else:
            self.branch = Branch(bw,
                                 use_acm=use_acm)

        self.use_clinic = use_clinic
        self.return_fc = return_fc
        if self.use_clinic:
            print("Use clinic data !")

        softmaxer = nn.Sigmoid() if num_classes == 2 else nn.Softmax()
        
        fc_in = 2*bw*4 + 4 if self.use_clinic else 2*bw*4

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
        if use_w_init and not pretrained:
            self.apply(weights_init_uniform)

    def forward(self, x1, x2=None, clinic=None, save=False):
        """
        [ Feature-level fusion ]
        Here, Radiomics Feauture + deep features
        o1 -> left features, o2 -> right features
        Each radiomic features are already extracted from the images
        
        """
        # Shared weights
        o1 = self.branch(x1) # left
        o2 = self.branch(x2) if x2 is not None else o1
        
        if save:
            self.left_value = o1
            self.right_value = o2
    
        feats = torch.cat([o1, o2, clinic], 1) if self.use_clinic else torch.cat([o1, o2], 1)
        
        if self.return_fc:
            return self.final(feats), feats
        else:
            return self.final(feats)


class OsteoSiameseDataset(torch.utils.data.Dataset):
    """Some Information about OsteoSiameseDataset"""
    def __init__(self, pair_list, transform, oai=False, multi=False, crop=False, triplet=False):
        super(OsteoSiameseDataset, self).__init__()

        self.transform = transform
        self.pair_list = pair_list
        self.oai = oai
        self.multi = multi
        self.crop = crop
        self.triplet = triplet

        if triplet:
            self.osteo_list = [x for x in self.pair_list if int(x[0].split('/')[-2]) == 1]
            self.normal_list = [x for x in self.pair_list if int(x[0].split('/')[-2]) == 0]

        import pickle as pkl

        with open('../Data/clinic_data/whole_clinic_final.pickle', 'rb') as f:
            snuh_clinic_data = pkl.load(f)

        with open('../Data/clinic_data/brmh_clinic_final.pickle', 'rb') as f:
            brmh_clinic_data = pkl.load(f)

        self.clinical_data = {**snuh_clinic_data, **brmh_clinic_data}

        if self.multi:
            with open('./multi_label.pkl', 'rb') as f:
                self.multi_label = pkl.load(f)

    def random_pick(self, current_subject, current_label, is_pos=True):
        target_label = current_label if is_pos else int(not(current_label))
        target_list = self.osteo_list if target_label == 1 else self.normal_list
        if not is_pos:
            return random.choice(target_list)
        else:
            target_list = [x for x in target_list if x[0].split('/')[-1].replace('_0.png', '') != str(current_subject)]
            return random.choice(target_list)

    def __getitem__(self, index):

        left, right = self.pair_list[index]

        # subject_name
        subject_name = left.split('/')[-1].replace('_0.png', '')
        clinic_data = self.clinical_data[subject_name]

        try:
            label = int(left.split('/')[-2])
            if self.triplet:
                positive_sample = self.random_pick(subject_name, label, is_pos=True)
                negative_sample = self.random_pick(subject_name, label, is_pos=False)

            if self.multi:
                label = self.multi_label[subject_name]

        except:
            label = int(np.random.randint(2))
            if self.multi:
                label = int(np.random.randint(3))

        def read_image(left, right):
            left_ = cv2.imread(left)[:, :, 0]
            right_ = np.fliplr(cv2.imread(right)[:, :, 0])

            left_ = (left_ - left_.min()) / (left_.max() - left_.min())
            right_ = (right_ - right_.min()) / (right_.max() - right_.min())

            left_ = left_[:int(left_.shape[0] * 0.9), :].astype(np.float32)
            right_ = right_[:int(left_.shape[0]), :].astype(np.float32)

            left_ = self.transform(left_)
            right_ = self.transform(right_)

            return left_, right_

        anchor_left, anchor_right = read_image(left, right)

        if self.triplet:
            postive_left, positive_right = read_image(*positive_sample)
            negative_left, negative_right = read_image(*negative_sample)

            data = dict(
                name = subject_name,
                image = [anchor_left, anchor_right],
                clinic = torch.Tensor(clinic_data).type(torch.FloatTensor),
                label = label,
                positive = [postive_left, positive_right],
                negative = [negative_left, negative_right]
            )

            return data
        else:
            data = dict(
                name = subject_name,
                image = [anchor_left, anchor_right],
                clinic = torch.Tensor(clinic_data).type(torch.FloatTensor),
                label = label
            )
            return data

    def __len__(self):
        return len(self.pair_list)


def train_test_split(pair_list, ratio=0.2, multi=False):

    if ratio == 0:
        return pair_list

    if not multi:
        
        minors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
        majors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(0)]
        
        test_minor, train_minor = minors[:int(len(minors) * ratio)], minors[int(len(minors) * ratio):]
        test_major, train_major = majors[:int(len(majors) * ratio)], majors[int(len(majors) * ratio):]

        return test_minor+test_major, train_minor+train_major

    elif multi:

        subject_name = lambda x : '_'.join(x.split('/')[-1].split('_')[:2])

        with open('./multi_label.pkl', 'rb') as f:
            multi_label = pkl.load(f)

        penia_subjects = [a for a, value in multi_label.items() if value==1]

        porosis = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
        normal = [pair for pair in pair_list if (pair[0].split('/')[-2] == str(0)) and (subject_name(pair[0]) not in penia_subjects)]
        penia = [pair for pair in pair_list if (pair[0].split('/')[-2] == str(0)) and (subject_name(pair[0]) in penia_subjects)]
        
        test_penia, train_penia = penia[:int(len(penia) * ratio)], penia[int(len(penia) * ratio):]
        test_normal, train_normal = normal[:int(len(normal) * ratio)], normal[int(len(normal) * ratio):]
        test_porosis, train_porosis = porosis[:int(len(porosis) * ratio)], porosis[int(len(porosis) * ratio):]

        return test_normal+test_penia+test_porosis, train_normal+train_penia+train_porosis

def random_over_sampling(pair_list, multi=False):
    if not multi:
        minors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
        majors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(0)]

        sampled = []

        for i in range(len(majors)//len(minors)):
            sampled.extend(random.sample(minors, len(minors)))

        return majors + sampled

    if multi:
        subject_name = lambda x : '_'.join(x.split('/')[-1].split('_')[:2])

        with open('./multi_label.pkl', 'rb') as f:
            multi_label = pkl.load(f)

        penia_subjects = [a for a, value in multi_label.items() if value==1]

        porosis = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
        normal = [pair for pair in pair_list if (pair[0].split('/')[-2] == str(0)) and (subject_name(pair[0]) not in penia_subjects)]
        penia = [pair for pair in pair_list if (pair[0].split('/')[-2] == str(0)) and (subject_name(pair[0]) in penia_subjects)]

        majors = penia

        whole = []

        for minors in [normal, porosis]:
            sampled = []
            for i in range(len(majors)//len(minors)):
                sampled.extend(random.sample(minors, len(minors)))

            print(len(sampled))
            whole.extend(sampled)
        
        print("normal {}, porosis {}, penia {}".format(len(normal), len(porosis), len(penia)))
        return majors + whole

def getPairList(root_dir, duplicate = False):
    images = sorted(glob(os.path.join(root_dir, '*/*.png')))
    images = sorted(glob(os.path.join(root_dir, '*.png'))) if len(images) == 0 else images # For test folder
    temp, pairs = dict(), dict()

    for img in images:
        subject = img.split('/')[-1].replace('.png', '')
        year, index, side = subject.split('_')

        try:
            temp['_'.join([year, index])].append(img)
            if len(temp['_'.join([year, index])]) == 2:
                pairs['_'.join([year, index])] = temp['_'.join([year, index])]

        except:
            temp['_'.join([year, index])] = [img]
            
    if duplicate:
        for key, value in temp.items():
            if len(value) == 1:
                pairs[key] = temp[key] * 2

    pair_list = list(pairs.values())
    return pair_list


def getExternalPairList(root_dir):
    images = sorted(glob(os.path.join(root_dir, '*.png')))
    temp, pairs = dict(), dict()

    for img in images:
        subject = img.split('/')[-1].replace('.png', '')
        index, side = subject.split('_')

        try:
            temp[index].append(img)
            if len(temp[index]) == 2:
                pairs[index] = temp[index]

        except:
            temp[index] = [img]

    pair_list = list(pairs.values())
    return pair_list

# data_dir = '/sdb1/share/ai_osteoporosis_brmh_png/dl/image/'
# pair_list = getExternalPairList(data_dir)

class LabelSmoothingLoss(nn.Module):
    """
        Label Smoothing Loss
        Code modified by Sang Wook Kim
    """
    
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        # pred size : B x class number
        # target size : B x 1
        pred = pred.log_softmax(dim=self.dim)
        target = torch.argmax(target, dim=1).long()
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def load_model(resume_file, model):
    from collections import OrderedDict
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    checkpoint = torch.load(resume_file)
    state_dict = checkpoint['state_dict']
    
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.lstrip('module.') # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def binary2multi(model, num_classes=3):
    new_final_comps = torch.nn.Sequential(
        model.final[0],
        torch.nn.Linear(in_features=256, out_features=num_classes, bias=True),
        torch.nn.Softmax()
    )
    model.final = new_final_comps
    return model

def multi2binary(model, num_classes=2):
    new_final_comps = torch.nn.Sequential(
        model.final[0],
        torch.nn.Linear(in_features=256, out_features=num_classes, bias=True),
        torch.nn.Sigmoid()
    )
    model.final = new_final_comps
    return model

def tf_learning(model, num_classes=2):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    new_block2 = model.branch.block2
    new_block2 = new_block2.apply(weights_init_uniform)
    new_final_comps = torch.nn.Sequential(
        model.final[0],
        torch.nn.Linear(in_features=256, out_features=num_classes, bias=True),
        torch.nn.Sigmoid()
    )
    model.final = new_final_comps
    model.branch.block2 = new_block2
    return model

if __name__ == "__main__":

    # first_channels = 32
    # BATCH_SIZE = 2
    # num_classes = 5
    # model = OsteoSiameseNet(bw=first_channels, drop=0.3, num_classes=num_classes)
    #
    # x1 = torch.Tensor(BATCH_SIZE, 1, 256, 256)
    # x2 = torch.Tensor(BATCH_SIZE, 1, 256, 256)
    #
    # y = model(x1, x2)
    #
    # print(y.size())

    def crop_bottom(arr, ratio=1.5, avg_h=1334):
        is_short = arr.shape[0] <= avg_h

        if is_short:
            return arr
        else:
            return arr[:int(arr.shape[1] * ratio), :]

    import matplotlib.pyplot as plt

    data_dir = '/sdb1/share/ai_osteoporosis_hip/real_final_png/dl_cropped/development_label/'
    brmh_dir = '/sdb1/share/ai_osteoporosis_hip/real_final_png/dl_cropped/brmh'

    data_list = glob(os.path.join(data_dir, '0', '*.png'))
    brmh_list = glob(os.path.join(brmh_dir, '*.png'))

    whole_shapes = []
    cropped_shapes = []
    for img in data_list[:5]:
        img_arr = cv2.imread(img)[:, :, 0]
        print(img_arr.max(), img_arr.min(), img_arr.mean())

        img_arr_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        print(img_arr_normed.max(), img_arr_normed.min(), img_arr_normed.mean())

        plt.subplot(1, 2, 1)
        plt.imshow(img_arr, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.imshow(img_arr_normed, cmap='gray')
        plt.show()

    for img in brmh_list[:5]:
        img_arr = cv2.imread(img)[:, :, 0]
        print(img_arr.max(), img_arr.min(), img_arr.mean())

        img_arr_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        print(img_arr_normed.max(), img_arr_normed.min(), img_arr_normed.mean())

        plt.subplot(1, 2, 1)
        plt.imshow(img_arr, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.imshow(img_arr_normed, cmap='gray')
        plt.show()

        # whole_shapes.append(img_arr.mean())

        # cropped_img_arr = crop_bottom(img_arr, )
        # break

    # print(np.array(whole_shapes), np.array(whole_shapes).mean())
