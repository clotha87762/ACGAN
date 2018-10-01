# -*- coding: utf-8 -*-
import torch
from torchvision.utils import *
from torchvision.transforms import *
from torchvision.datasets import *

import os
import argparse

import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()

parser.add_argument('--batch' , dest = 'batch' , type = int , default = 32)

parser.add_argument('--epoch' , dest = 'epoch' , type = int , default = 200)
parser.add_argument('--imsize', dest = 'imsize' , type = int , default = 32 , help = 'image size')
parser.add_argument('--gfdim' , dest = 'gfdim' , type = int , default = 16 )
parser.add_argument('--deconv' , dest = 'deconv' type = bool , default = True )
parser.add_argument('--dfdim' , dest = 'dfdim' , type = int , default = 16 )
parser.add_argument('--in_dim', dest = 'in_dim' , type = int , default = 3  ,help = 'input image channel')
parser.add_argument('--out_dim' , dest = 'out_dim' , type = int , default = 3 )
parser.add_argument('--lr' , dest = 'lr' , type = float , default = 0.0002)
parser.add_argument('--num_class' , dest = 'num_class' , type = int , default = 10 )
parser.add_argument('--beta1' , dest = 'beta1' , type = float , default = 0.5 , help = 'beta1 of Adam optimizer')
parser.add_argument('--dim_embed' , dest = 'dim_embed' , type = int , default = 100 , help = 'the dim of the embedded vector' )

parser.add_argument('--sn' , dest = 'sn' , type = bool , default = True , help = ' Use spectral normalization or not')

parser.add_argument('--g_kernel' , dest = 'g_kernel' , type = int , default = 3 , help = 'kernel size of generator conv2d')
parser.add_argument('--d_kernel' , dest = 'd_kernel' , type = int , default = 3 , help = 'kernel size of discriminator conv2d')

parser.add_argument('--use_gpu', dest = 'gpu' , type = bool , default = True)
parser.add_argument('--gpu_idx', dest = 'gpu_idx' , default = '0' )


parser.add_argument('--data_dir' , dest = 'data_dir' , default = './Dataset')
parser.add_argument('--ckpt_dir' , dest = 'ckpt_dir' , default = './Checkpoint')
parser.add_argument('--log_dir' , dest = 'log_dir' , default = './Logs')
parser.add_argument('--sample_dir' , dest = 'sample_dir' , default = './Sample')
parser.add_argument('--test_dir' , dest = 'test_dir' , default = './Test')
parser.add_argument('--dataset', dest = 'dataset' , default = 'cifar10')

parser.add_argument('--seed', dest = 'seed' , default = None)


args = parser.parse_args()


def main(_):
    
    if not os.path.exists(data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    if args.use_gpu :
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    
    if args.seed is not None :
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print('Using seed' + str(seed))
    
    if not torch.cuda.is_available():
        if args.gpu:
            print('Cuda is not available, please disable gpu')
            exit()
    
    if len(args.gpu_idx) > 1 :
        multi_gpu = True
    else :
        multi_gpu = False
    
    
    
    
        

    
if __name__ == '__main__' :
    main()



