import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchtoolbox.tools import mixup_data, mixup_criterion

from torch.utils.tensorboard import SummaryWriter

from Siamese import OsteoSiameseDataset, OsteoSiameseNet, train_test_split, getPairList, getExternalPairList, random_over_sampling, load_model, tf_learning
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training Code')
parser.add_argument('data', metavar='DIR', default='/sdb1/share/ai_osteoporosis_hip/real_final_png/dl_pair_whole/development/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet121)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--height_size', default=224, type=int, metavar='N',
                    help='height of input image (default : 224)')
parser.add_argument('--width_size', default=224, type=int, metavar='N',
                    help='width of input images (default : 224)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epoch_cp', default=30, type=int,
                    help='learning rate step epoch ( default: 30 )')
parser.add_argument('--val_ratio', default=0.2, type=float, dest='val_ratio',
                    help='Validation ratio (default: 0.2)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--fc_save_path', default='./', type=str,
                    help='path to fc layer value')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--multi', action='store_true', default=False,
                    help='Multi-class classification ( default: False )')
parser.add_argument('--ext', dest='ext', action='store_true',
                    help='evaluate model on external set')
parser.add_argument('--duplicate', dest='duplicate', action='store_true',
                    help='duplicate only one-sided images')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--mixup', dest='mixup', default=0, type=float,
                    help='Use mixup data augmentation')
parser.add_argument('--save_fc', dest='save_fc', action='store_true',
                    help='save fc layer value')
parser.add_argument('--use_resnet', dest='use_resnet', action='store_true',
                    help='use resnet18 backbone architecture (not pretrained)')
parser.add_argument('--curriculum', default='', type=str, metavar='PATH',
                    help='path to weight file')
parser.add_argument('--label_smoothing', dest='label_smoothing', action='store_true',
                    help='Label smoothing')
parser.add_argument('--smoothing_param', default=0.1, type=float, metavar='M',
                    help='label smoothing hyperparameter (default: 0.1)')
parser.add_argument('--save_fc_only', dest='save_fc_only', action='store_true',
                    help='only for save')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--version', default='v0', type=str,
                    help='version (default: v0)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.curriculum:
        if os.path.isfile(args.curriculum):
            print("=> using pre-trained model '{}'".format(args.curriculum))
            try:
                # binary -> multi
                model = OsteoSiameseNet(bw=32,
                                        drop=0.3,
                                        num_classes=2,
                                        use_resnet=args.use_resnet)
                loaded_model = load_model(args.curriculum, model)
                model = tf_learning(loaded_model, 3)

            except:
                # multi -> binary
                model = OsteoSiameseNet(32, 0.3, 3, True)
                loaded_model = load_model(args.curriculum, model)
                model = tf_learning(loaded_model, 2)
        else:
            print("=> There is no file called {}".format(args.curriculum))
    else:
        print("=> creating model '{}'".format('Osteo-Siamese Network'))
        # model = models.__dict__[args.arch]()
        if args.multi:
            model = OsteoSiameseNet(bw=32, drop=0.3, num_classes=3, use_resnet=args.use_resnet)
        else:
            model = OsteoSiameseNet(bw=32,
                                    drop=0.3,
                                    num_classes=2,
                                    use_resnet=args.use_resnet)
        
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(model)

    if args.label_smoothing:
        from Siamese import LabelSmoothingLoss
        if args.multi:
            criterion = LabelSmoothingLoss(classes=3, smoothing=args.smoothing_param)
        else:
            criterion = LabelSmoothingLoss(classes=2, smoothing=args.smoothing_param)
    else:
        # define loss function (criterion) and optimizer
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Tensorboard Setting
    TB = TensorBoard(log_dir=os.path.join('./runs/', '{}'.format(args.version)))

    # Data loading code
    image_dir = os.path.join(args.data)

    if args.ext:
        pair_list = getExternalPairList(image_dir)
    else:
        pair_list = getPairList(image_dir, duplicate=args.duplicate)

    print(len(pair_list))

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.height_size, args.width_size)),
            transforms.ToTensor()
        ])

    if args.save_fc_only:
        whole_dataset = OsteoSiameseDataset(pair_list, transform)

        whole_loader = torch.utils.data.DataLoader(
            whole_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        validate(whole_loader, model.module.cuda(), criterion, 1, args, TB)
        return

    val_list, train_list = train_test_split(pair_list, ratio=args.val_ratio, multi=args.multi)
    print(len(train_list), len(val_list))
    
    train_list = random_over_sampling(train_list, multi=args.multi)

    train_dataset = OsteoSiameseDataset(train_list, transform, oai=False, multi=args.multi)
    validation_dataset = OsteoSiameseDataset(val_list, transform, oai=False, multi=args.multi)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model.module.cuda(), criterion, 1, args, TB)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, TB)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, TB)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.version)


def labelshaper(labels, multi=False):
    if not multi:
        return torch.Tensor([[0, 1] if label == 1 else [1, 0] for label in labels]).to(torch.float)
    elif multi:
        shaped_labels = []
        for label in labels:
            dummy = [0, 0, 0]
            dummy[label] = 1
            shaped_labels.append(dummy)
        return torch.Tensor(shaped_labels).to(torch.float)


def train(train_loader, model, criterion, optimizer, epoch, args, TB):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (images, target) in enumerate(train_loader):
    for i, data in enumerate(train_loader):

        images = data['image']
        name = data['name']
        target = data['label']

        target = labelshaper(target, args.multi)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.mixup > 0:
            left_, right_ = images
            mixed_images, labels_a, labels_b, lam = mixup_data(torch.cat((left_, right_), dim=1), target, args.mixup)

            left_ = mixed_images[:, 0].unsqueeze(1)
            right_ = mixed_images[:, 1].unsqueeze(1)
            images = [left_, right_]

        if args.gpu is not None:
            left_, right_ = images
            left_, right_ = left_.cuda(args.gpu, non_blocking=True), right_.cuda(args.gpu, non_blocking=True)

            if args.mixup > 0:
                mixed_images, labels_a, labels_b, lam = mixup_data(torch.cat((left_, right_), dim=1), target, args.mixup)

                left_ = mixed_images[:, 0].unsqueeze(1)
                right_ = mixed_images[:, 1].unsqueeze(1)

            images = [left_, right_]

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(*images)

        if args.mixup > 0:
            loss = mixup_criterion(criterion, output, labels_a.cuda(), labels_b.cuda(), lam)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # save for calculate auc score => One class problem
        if i == 0:
            targets, outputs = target, output
        elif i != 0:
            targets = torch.cat((targets, target))
            outputs = torch.cat((outputs, output))

    # TensorBoard update
    TB.update('Accuracy/Train', top1.avg, epoch)
    TB.update('Loss/Train', losses.avg, epoch)
    TB.update('AUC/Train', auc(outputs, targets), epoch)


def validate(val_loader, model, criterion, epoch, args, TB):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    names = []

    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in enumerate(val_loader):
        for i, data in enumerate(val_loader):

            left_, right_ = data['image']
            name = data['name']
            target = data['label']

            target = labelshaper(target, args.multi)

            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            left_, right_ = left_.cuda(), right_.cuda()
            target = target.cuda()

            # compute output
            output = model(left_, right_, save=args.save_fc)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # save fc layer value to local folder
            if args.save_fc:
                left_vector = model.left_value.detach().cpu().numpy()
                right_vector = model.right_value.detach().cpu().numpy()
                save_fc(left_vector, right_vector, name, args.fc_save_path)

            if i % args.print_freq == 0:
                progress.display(i)

            # save for calculate auc score => One class problem
            if i == 0:
                targets, outputs = target, output
            elif i != 0:
                targets = torch.cat((targets, target))
                outputs = torch.cat((outputs, output))

            names.extend(name)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1) + 'AUC : {}'.format(auc(outputs, targets)))

    confidence_save(names, outputs, './', args.version)
    print('confidence file saved')

    # TensorBoard update
    TB.update('Accuracy/Val', top1.avg, epoch)
    TB.update('Loss/Val', losses.avg, epoch)
    TB.update('AUC/Val', auc(outputs, targets), epoch)

    return top1.avg


def save_checkpoint(state, is_best, version):
    filename = 'checkpoint_{}.pth.tar'.format(version)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(version))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # print(fmtstr, self.__dict__)
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TensorBoard():
    def __init__(self, log_dir):
        super(TensorBoard, self).__init__()
        self.tb = SummaryWriter(log_dir=log_dir)

    def update(self, field, value, epoch):
        self.tb.add_scalar(field, value, epoch)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_cp))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, target = torch.max(target, 1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]


def auc(output, target):
    with torch.no_grad():
        return roc_auc_score(target.cpu().numpy(), output.cpu().numpy(), multi_class='ovo', average='macro')
    # if not args.multi:
    #     with torch.no_grad():
    #         return roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
    # else:
    #     with torch.no_grad():
        # return roc_auc_score(target.cpu().numpy(), output.cpu().numpy(), multi_class='ovo', average='macro')


def save_fc(left_vector, right_vector, names, path):
    for lv, rv, name in zip(left_vector, right_vector, names):
        np.save(os.path.join(path, name + '_0.npy'), lv)
        np.save(os.path.join(path, name + '_1.npy'), rv)


def confidence_save(names, outputs, save_folder, version):
    softmax = torch.nn.Softmax(dim=1)
    predict_labels = torch.argmax(outputs, 1).detach().cpu().numpy()
    predict_proba = softmax(outputs).detach().cpu().numpy()
    ext_result = np.hstack([np.expand_dims(predict_labels, 1), predict_proba])

    if predict_proba.shape[1] == 3:
        external_result = pd.DataFrame(ext_result, index=names,
                                       columns=['label',
                                                'confidence score for class 0',
                                                'confidence score for class 1',
                                                'confidence score for class 2'])
    else:
        external_result = pd.DataFrame(ext_result, index=names,
                                       columns=['label',
                                                'confidence score for class 0',
                                                'confidence score for class 1'])

    external_result.to_csv(os.path.join(save_folder, 'siamese_mlp_{}.csv'.format(version)))


if __name__ == '__main__':
    main()