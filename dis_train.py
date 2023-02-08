import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from multiscaleloss import multiscaleEPE, realEPE
import models
import datasets
import numpy as np
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


dataset_names = sorted(name for name in datasets.__all__)
best_EPE = -1
n_iter = 0

# device = torch.device("cpu")
# os.environ[‘CUDA_VISIBLE_DEVICES’] = ‘0,1’
world_size = torch.cuda.device_count()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='U-DICNet Training on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--arch', default='U_DICNet', choices=['StrainNet_f', 'U_DICNet', 'U_StrainNet_f'],
                    help='network selection')
parser.add_argument('--train_dataset_root', '-trr', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--test_dataset_root', '-ter', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--solver', default='adam', choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240],
                    metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
# parser.add_argument('--multiscale-weights', '-w', default=[0.12, 0.04, 0.08, 0.01, 0.02], type=float, nargs=5,
#                     help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
#                     metavar=('W2', 'W3', 'W4', 'W5', 'W6'))

parser.add_argument('--multiscale-weights', '-w', default=[0.01, 0.02, 0.05, 0.08, 0.24], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))  # 0.02, 0.02, , 'W5', 'W6'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def setup(rank, world_size):
    # initialize the process group

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    # destroy the process

    torch.distributed.destroy_process_group()


def reduce_mean(tensor, nprocs):
    # communication of different processes

    rt = tensor.clone()
    # calculate the average value
    torch.distributed.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    global world_size
    # start the multi-processing
    mp.spawn(numworker,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def numworker(rank, world_size):

    global args, best_EPE

    args = parser.parse_args()

    save_path = './{}_network_data/{}epochs,b{},lr{}'.format(
        args.arch,
        args.epochs,
        args.batch_size,
        args.lr)
    if rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    setup(rank, world_size)

    # train_dataset_root = '/home/lanshihai/program code/Lan/1_restart_train/128img_r1.2_interp100/'
    # test_dataset_root = '/home/lanshihai/program code/Lan/1_restart_train/img128_r1.2_test_interp10/'
    # pre_train = './pretrained_model/model_best.pth.tar' # model_best.pth.tar'#

    # network_data = torch.load(pre_train)
    # print("=> using pre-trained model ")

    # dataset
    train_set, test_set = datasets.__dict__['speckle_dataset'](args.train_dataset_root, args.test_dataset_root, args.arch)

    # dataset sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=int(args.batch_size/world_size),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(args.batch_size/world_size),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # create model
    if args.pretrained:
        # using pre-trained model
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model ")
        best_EPE = network_data['best_EPE']
    else:
        network_data = None
        print("=> creating new model ")

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    # network_data = torch.load(pre_train)
    # print("=> using pre-trained model ")

    # choose the model
    model = models.__dict__['U_DICNet'](network_data).to(rank)  # , drop=False)

    # training parameters
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    # Distribute model in different GPUs
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    cudnn.benchmark = True

    # selection of the optimizer
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):

        # start the training of the model
        train_loss, train_EPE = train(train_loader, model, optimizer, scheduler, rank)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        with torch.no_grad():
            # validation of the model
            EPE = validate(test_loader, model, rank)
            test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        if rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
            }, is_best, save_path)
    if world_size > 1:
        cleanup()


def train(train_loader, model, optimizer, scheduler, rank):

    global args, world_size

    losses = AverageMeter()
    flow_EPEs = AverageMeter()
    epoch_size = len(train_loader)
    model.train()

    for i, (input_img, target) in enumerate(train_loader):
        # input images and the real displacement
        target = target.to(rank)
        input_img = input_img.to(rank)

        # calculate the output
        output = model(input_img)

        # calculate the loss function
        loss = multiscaleEPE(output, target, rank, weights=args.multiscale_weights)

        # calculate the average endpoint error except for the edge
        flow_EPE = realEPE(output[0], target, rank)

        if world_size > 1:
            # process synchronization
            torch.distributed.barrier()

            # calculate the average loss in different process
            reduced_loss = reduce_mean(loss, world_size)

            # calculate the average flow_EPE in different process
            reduced_flow_EPE = reduce_mean(flow_EPE, world_size)

            # update the record
            losses.update(reduced_loss.item(), target.size(0))
            flow_EPEs.update(reduced_flow_EPE.item(), target.size(0))
        else:
            losses.update(loss.item(), target.size(0))
            flow_EPEs.update(flow_EPE.item(), target.size(0))

        optimizer.zero_grad()
        # parameters update,  learning rate update
        loss.backward()
        optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率

        if i >= epoch_size:
            break
    # return the loss and the endpoint error
    return losses.avg, flow_EPEs.avg


def validate(val_loader, model, rank):

    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_img, target) in enumerate(val_loader):
        target = target.to(rank)
        input_img = input_img.to(rank)

        # compute output
        output = model(input_img)

        # record EPE
        flow2_EPE = realEPE(output, target, rank)

        # process synchronization
        if world_size > 1:
            torch.distributed.barrier()

            reduced_flow2_EPE = reduce_mean(flow2_EPE, world_size)
            flow2_EPEs.update(reduced_flow2_EPE.item(), target.size(0))
        else:
            flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
