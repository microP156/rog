import argparse
from SSP_utils import Local_Worker, Parameter_Server
from new_ROG_utils import ROG_Local_Worker, ROG_Parameter_Server, layer_unit
from DEFSGDM.DEFSGDM import DEFSGDM_server, DEFSGDM_worker
from DEFSGDM.ParameterWrapper import Parameter
import os
import random
import time
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from utils.dataloaders import *

import multiprocessing as mp

from ConvMLP import ConvMLP 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
# parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
#                     choices=DATA_BACKEND_CHOICES)
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     # choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-E', default=5, type=int, metavar='N',
                    help='local update')
parser.add_argument('--idx-start', default=0, type=int, metavar='N',
                    help='the start idx of dataset of this worker')
parser.add_argument('--idx-num', default=10, type=int, metavar='N',
                    help='dataset size for this worker')
parser.add_argument('--chkpt-dir', default='.', type=str, metavar='PATH',
                    help='dir for runtime chkpt')
parser.add_argument('--wnic-name', default='wlan0', type=str,
                    help='wnic name to get tc bandwidth')
parser.add_argument('--chkpt-rank', default=1, type=int, 
                    help='The worker to checkpoint')
parser.add_argument('--FLOWN-enable', dest='FLOWN_enable', action='store_true',
                    default=False, help='FLOWN_enabled')
parser.add_argument('--SSP-enable', dest='SSP_enable', action='store_true',
                    default=False, help='SSP_enabled')
parser.add_argument('--ROG-enable', dest='ROG_enable', action='store_true',
                    default=False, help='ROG_enabled')
parser.add_argument('--compression-enable', dest='compression_enable',
                    action='store_true', default=False)
                    
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:13456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')
parser.add_argument('--threshold', default=0, type=int, metavar='threshold',
                    help='staleness threshold')
parser.add_argument('--noise-type', type=str, default='image_blur',
                    help='mode for learning rate decay')
ps_ip = "10.42.0.1"
ps_port = 12345


def main():
    global args, best_prec1
    args = parser.parse_args()
    # args.seed = int(time.time())
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # random.seed(int(time.time()))
    # torch.manual_seed(int(time.time()))

    args.distributed = args.world_size > 1

    if args.distributed and not args.compression_enable:
        print(f"initializing process group {args.dist_backend}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    print("=> creating model")
    # model = models.__dict__[args.arch](width_mult=args.width_mult)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    idx_num_train = args.idx_num
    idx_num_test = 0
    convmlp = ConvMLP(args.rank - 1, batch_size=args.batch_size,
                      num_shards=args.world_size - 1,
                      num_client=args.world_size - 1)
    model, optimizer, lr_scheduler, train_dl, test_dl, train_loss_fn, mixup_fn = convmlp.init()
    if args.ROG_enable:
        layer_info=[]
        start_idx=0
        start_pos=0
        for p in model.parameters():
            each_layer_rows,start_idx,start_pos=layer_unit(p,start_idx,start_pos)
            layer_info.append(each_layer_rows) 
    model = model.to(device)
    if args.compression_enable:
        if args.rank == 0:
            optimizer = DEFSGDM_server(params=model.parameters(), worker_num=args.world_size-1, device=device)
        else:
            del optimizer, lr_scheduler
            # lower learning rate
            optimizer = DEFSGDM_worker(params=model.parameters(), device=device, lr=1e-6)
            lr_scheduler = MultiStepLR(optimizer, milestones=[800,1600,2400,3200], gamma=0.5)

    criterion = train_loss_fn.to(device)

    print(f'batch_size: {args.batch_size}; train_idx_num: {idx_num_train}; test_idx_num: {idx_num_test}; local_update: {args.E}')
    cudnn.benchmark = True
    cudnn.enabled = True

    print(f"arg FLOWN_enable {args.FLOWN_enable}")
    print(f"arg compression_enable (DEFSGDM) {args.compression_enable}")
    print(f"arg ROG_enable {args.ROG_enable}")
    if args.ROG_enable:
        communication_library = "rog"
    elif not args.compression_enable:
        communication_library = "gloo"
    else:
        communication_library = "tcp"
    if args.rank == 0:
        if args.ROG_enable:
            parameter_server = ROG_Parameter_Server(args, ps_ip, ps_port, args.world_size-1, args.threshold, model, layer_info,optimizer, communication_library, device, args.batch_size, args.chkpt_dir, compression_enable=args.compression_enable)
        else:
            parameter_server = Parameter_Server(args, ps_ip, ps_port, args.world_size-1, args.threshold, model, optimizer, communication_library, device, args.batch_size, args.chkpt_dir, FLOWN_enable=args.FLOWN_enable, compression_enable=args.compression_enable)
        del train_dl
    else:
        if args.ROG_enable:
            local_worker = ROG_Local_Worker(args, model, layer_info,ps_ip, ps_port, criterion, communication_library, train_dl, test_dl, device, lr_scheduler, optimizer, mixup_fn, compression_enable=args.compression_enable)
        else:
            local_worker = Local_Worker(args, model, ps_ip, ps_port, criterion, communication_library, train_dl, test_dl, device, lr_scheduler, optimizer, mixup_fn, compression_enable=args.compression_enable)

    if args.rank != 0:
        start_time = time.time()
        epoch_time = []
        for epoch in range(args.start_epoch, args.epochs):
            if hasattr(local_worker.lr_scheduler, '_get_lr'):
                lr_list = local_worker.lr_scheduler._get_lr(epoch)
            else:
                lr_list = local_worker.lr_scheduler.get_last_lr()
            lr = sum(lr_list) / len(lr_list)
            print('\nEpoch: [%d | %d] learning rate: %.2E' % (epoch + 1, args.epochs, lr))

            num_updates = epoch * len(train_dl)
            train_loss, num_updates = local_worker.train(epoch, start_time, num_updates, lr, local_update=args.E)
            epoch_time.append(time.time() - start_time)
        # print(f'fine-tuned accuracy on: {correct_rate}')
        print("End time of each epoch:")
        for i, e in enumerate(epoch_time):
            print(f"Epoch {i}: {e}")
        local_worker.terminate()
    print("Whole thread terminated")

if __name__ == '__main__':
    main()
