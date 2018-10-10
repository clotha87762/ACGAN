# -*- coding: utf-8 -*-
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader 
import file
import model

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()

parser.add_argument('--batch' , dest = 'batch' , type = int , default = 32)
parser.add_argument('--phase' , dest = 'phase' , default = 'train')

parser.add_argument('--epoch' , dest = 'epoch' , type = int , default = 200)
parser.add_argument('--imsize', dest = 'imsize' , type = int , default = 32 , help = 'image size')
parser.add_argument('--gfdim' , dest = 'gfdim' , type = int , default = 16 )
parser.add_argument('--deconv' , dest = 'deconv', type = bool , default = True )
parser.add_argument('--dfdim' , dest = 'dfdim' , type = int , default = 16 )

parser.add_argument('--in_dim', dest = 'in_dim' , type = int , default = 3  ,help = 'input image channel')
parser.add_argument('--out_dim' , dest = 'out_dim' , type = int , default = 3 )
parser.add_argument('--lr' , dest = 'lr' , type = float , default = 0.0002)

parser.add_argument('--num_class' , dest = 'num_class' , type = int , default = 10 )
parser.add_argument('--beta1' , dest = 'beta1' , type = float , default = 0.5 , help = 'beta1 of Adam optimizer')
parser.add_argument('--dim_embed' , dest = 'dim_embed' , type = int , default = 110 , help = 'the dim of the embedded vector' )

parser.add_argument('--sn' , dest = 'sn' , type = bool , default = True , help = ' Use spectral normalization or not')

parser.add_argument('--g_kernel' , dest = 'g_kernel' , type = int , default = 4 , help = 'kernel size of generator conv2d')
parser.add_argument('--d_kernel' , dest = 'd_kernel' , type = int , default = 4 , help = 'kernel size of discriminator conv2d')

parser.add_argument('--gpu', dest = 'gpu' , type = bool , default = True)
parser.add_argument('--gpu_idx', dest = 'gpu_idx' , default = '0' )


parser.add_argument('--data_dir' , dest = 'data_dir' , default = './Dataset')
parser.add_argument('--ckpt_dir' , dest = 'ckpt_dir' , default = './Checkpoint')
parser.add_argument('--log_dir' , dest = 'log_dir' , default = './Logs')
parser.add_argument('--sample_dir' , dest = 'sample_dir' , default = './Sample')
parser.add_argument('--test_dir' , dest = 'test_dir' , default = './Test')
parser.add_argument('--dataset', dest = 'dataset' , default = 'cifar10')

parser.add_argument('--sample_freq', dest = 'sample_freq' ,type = int , default = 500)
parser.add_argument('--save_freq', dest = 'save_freq' ,type = int , default = 500)



parser.add_argument('--seed', dest = 'seed' , default = None)
parser.add_argument('--worker', dest = 'worker' , type = int ,default = 4)
parser.add_argument('--wgan' , dest = 'wgan' , type = bool , default = False , help = 'Use WGAN training loss and strategy or not')
parser.add_argument('--clip', dest = 'clip' , type = float , default = 0.01 , help='clip  value of wgan')


parser.add_argument('--gp' , dest = 'gp' , type = bool , default = False , help ='If true at the same time with wgan, use wgan-gp')
parser.add_argument('--gp_weight' , dest = 'gp_weight' , type = float , default = 10.0)

parser.add_argument('--aux_weight' , dest = 'aux_weight', type = float , default = 1.0 , help = 'loss weight of auxiliary buffer')
parser.add_argument('--l_smooth' , dest = 'l_smooth' , type = bool , default = False , help = 'Label smoothing of GAN or not')

parser.add_argument('--run_name' , dest = 'run_name' , default = 'test' )

parser.add_argument('--sample_idx' , dest = 'sample_idx' ,type = int, default = None , help = 'sample index for evaluation in test phase')

args = parser.parse_args()


def weights_init(model):
    name = model.__class__.__name__
    if name.find('Conv') != -1 :
        model.weight.data.normal_(0.0 ,0.02)
    elif name.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
        

def sample_generator( generator , noise , label , step):
    
    fake = generator(noise , label).detach().cpu()
    vutils.save_image( fake , sample_path + '/%d.png' % step , nrow=8, normalize=True)

def test_generator( generator , noise , label , nrow ):
    
    # file name is sample index
    fake = generator(noise , label).detach().cpu()
    vutils.save_image( fake , test_path + '/%d.png' % args.sample_idx , nrow=nrow, normalize=True)
    

def train():
    
    
    if len(args.gpu_idx) > 1 :
        multi_gpu = True
        gpu_list = [ int(i) for i in args.gpu_idx.split(',')]
    else :
        multi_gpu = False
    
    writer = SummaryWriter(log_dir = log_path)
    
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.data_dir,download = True , 
                                    transform = transforms.Compose([
                                    transforms.Resize(args.imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
                                    ]
                                    )
                                   )
    elif args.dataset == 'mnist':
        dataset = datasets.MNIST(args.data_dir , train=True, download=True,
                       transform=transforms.Compose([
                            transforms.Scale( args.imsize ),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
        args.in_dim = 1
        args.out_dim = 1
    else :
        dataset = datasets.ImageFolder(args.data_dir , 
                                    transform = transforms.Compose([
                                    transforms.Resize(args.imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))]
                                    )
                                   )
    
    dataloader = torch.utils.data.DataLoader( dataset , batch_size = args.batch , \
                                             shuffle = True , num_workers = args.worker)
    
    #device = torch.device()
    
    generator = model.Generator(args)
    discriminator = model.Discriminator(args)
        
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    
    gan_criterion = nn.BCELoss()
    aux_criterion = nn.CrossEntropyLoss()
    
    
    #input_noise = torch.from_numpy( np.random.normal(0,1,[args.batch , args.dim_embed]) )
    #input_label = torch.from_numpy( np.random.randint(0,args.num_class, [args.batch,1 ]) )
    
    
    
   
    
    if args.l_smooth:
        # training strategy stated in improved GAN
        real_label = 0.9
        fake_label = 0.1
    else :
        real_label = 1.0
        fake_label = 0.0
        
    step = 0
    
    
    
    if args.gpu:

        # acutally do nothing?  because bce and cce don't have paramters
        gan_criterion = gan_criterion.cuda()
        aux_criterion = aux_criterion.cuda()
        
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        
        if multi_gpu:
            print('multi gpu')
            generator = nn.DataParallel(generator , device_ids = gpu_list)
            discriminator = nn.DataParallel(discriminator , device_ids = gpu_list)
    
    opt_d = optim.Adam(discriminator.parameters() , lr = args.lr , betas=(args.beta1 , 0.999) )
    opt_g = optim.Adam(generator.parameters() , lr = args.lr , betas=(args.beta1, 0.999) )
    
    if os.path.isfile(os.path.join(ckpt_path,args.run_name+'.ckpt')):
        print('found ckpt file' + os.path.join(ckpt_path,args.run_name+'.ckpt'))
        ckpt = torch.load(os.path.join(ckpt_path,args.run_name+'.ckpt'))
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        opt_d.load_state_dict(ckpt['opt_d'])
        opt_g.load_state_dict(ckpt['opt_g'])
        step = ckpt['step']
    
    
    for i in range(args.epoch):
        
        for j , data in enumerate(dataloader):
            
            images , labels = data[0] , data[1]
            
            batch = images.shape[0]
            
            
            input_noise = torch.from_numpy( np.random.normal(0,1,[batch , args.dim_embed]).astype(np.float32) )
            input_label = torch.from_numpy( np.random.randint(0,args.num_class, [batch ]) )
            
            
            input_noise =  np.random.normal(0,1,[batch , args.dim_embed]).astype(np.float32)
            class_onehot = np.zeros((batch, args.num_class))
            class_onehot[np.arange(batch), input_label] = 1
            input_noise[np.arange(batch), : args.num_class] = class_onehot[np.arange(batch)]
            input_noise = torch.from_numpy(input_noise)
            
            
            real_target = torch.full((batch,1) , real_label)
            fake_target = torch.full((batch,1) , fake_label)
            aux_target =  torch.autograd.Variable(labels)
             
            if args.gpu:
                input_noise = input_noise.cuda()
                input_label = input_label.cuda()
                images = images.cuda()
                labels = labels.cuda()
                real_target = real_target.cuda()
                fake_target = fake_target.cuda()
                aux_target = aux_target.cuda()
                
            # train generator
            # 好像不call也沒關係
            opt_g.zero_grad()
            
            fake = generator(input_noise , input_label)
            gan_out_g , aux_out_g = discriminator(fake)
            
            if args.wgan:
                gan_loss_g = -torch.mean(gan_out_g)
            else:
                gan_loss_g = gan_criterion(gan_out_g , real_target )
                
            aux_loss_g = aux_criterion(aux_out_g, input_label)
            
            g_loss =  (gan_loss_g +  args.aux_weight * aux_loss_g) * 0.5
            g_loss.backward()
            
            #opt_g.step()
            opt_g.step()
            
            # train discriminator with real samples
            opt_d.zero_grad()
            
            gan_out_r , aux_out_r = discriminator(images)
            
            if args.wgan:
                gan_loss_r = -torch.mean(gan_out_r)
            else:
                gan_loss_r = gan_criterion(gan_out_r , real_target )
            
            
            aux_loss_r = aux_criterion(aux_out_r , aux_target)
            d_real_loss = (gan_loss_r + args.aux_weight *  aux_loss_r) / 2.0
            
            
            # train discriminator with fake samples
            #fake = generator(input_noise , input_label).detach()
            gan_out_f , aux_out_f = discriminator(fake.detach())
            
            if args.wgan:
                gan_loss_f = torch.mean(gan_out_f)
            else:
                gan_loss_f = gan_criterion(gan_out_f , fake_target)
                

            aux_loss_f = aux_criterion(aux_out_f, input_label)
            d_fake_loss =( gan_loss_f +  args.aux_weight * aux_loss_f ) / 2.0

            
            
            if args.wgan and args.gp:
                gp = model.gradient_penalty(discriminator, images, fake , args.num_class , args.gpu)
                d_loss = 0.5 * (d_real_loss + d_fake_loss) + args.gp_weight*gp

            else:
                d_loss = (d_real_loss + d_fake_loss) / 2.0
                
            d_loss.backward()
            
            opt_d.step()
            
            
            
            step = step + 1
            if step % 100 == 0 : 
            	writer.add_scalar('losses/g_loss' , g_loss , step)
            	writer.add_scalar('losses/d_loss' , d_loss , step)
            	grid = vutils.make_grid(fake.detach() ,  normalize=True )
            	writer.add_image('generated', grid , step)
            
            if args.wgan and not args.gp :
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip , args.clip)
            
            if step % args.save_freq == 0 :
                torch.save({
                        'step' : step,
                        'generator' : generator.state_dict(),
                        'discriminator' : discriminator.state_dict(),
                        'opt_d' : opt_d.state_dict(),
                        'opt_g' : opt_g.state_dict()
                        }, os.path.join(ckpt_path,args.run_name+'.ckpt'))
            
            if step % args.sample_freq == 0 :
                sample_generator(generator , input_noise , input_label , step)
            
            pred = np.concatenate([aux_out_r.data.cpu().numpy(), aux_out_f.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), input_label.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % \
                   (i, args.epoch , j, len(dataloader),
                    d_loss.item(), 100.0 * d_acc,
                    g_loss.item()))
       



def test():
    
    nrow = 4  # output a 4 * 4 grid
    
    if args.dataset == 'mnist' :
        args.in_dim = 1
        args.out_dim = 1
    
    generator = model.Generator(args)
    
        
    if args.gpu:
        # no need to use multi gpu in testing phase
        generator = generator.cuda()
        
    assert os.path.isfile(os.path.join(ckpt_path,args.run_name+'.ckpt'))
    
    print('found ckpt file' + os.path.join(ckpt_path,args.run_name+'.ckpt'))
    ckpt = torch.load(os.path.join(ckpt_path,args.run_name+'.ckpt'))
    generator.load_state_dict(ckpt['generator'])
        
    #input_noise = torch.from_numpy( np.random.normal(0,1,[ nrow**2 , args.dim_embed]).astype(np.float32) )
    
    
    
    if args.sample_idx == None:
        input_label = torch.from_numpy( np.random.randint(0,args.num_class, [ nrow**2 ]) )
    else :
        input_label = torch.from_numpy( np.array([ np.int(args.sample_idx) for i in range(nrow**2)]))
    
    input_noise =  np.random.normal(0,1,[nrow**2 , args.dim_embed]).astype(np.float32)
    class_onehot = np.zeros((nrow**2, args.num_class))
    class_onehot[np.arange(nrow**2), input_label] = 1
    input_noise[np.arange(nrow**2), : args.num_class] = class_onehot[np.arange(nrow**2)]
    input_noise = torch.from_numpy(input_noise)
    
    
    if args.gpu:
        input_noise = input_noise.cuda()
        input_label = input_label.cuda()
        
    generator.zero_grad()
    #fake = generator(input_noise , input_label)
    
    test_generator(generator , input_noise , input_label , nrow )
            
          
if __name__ == '__main__' :
    log_path = os.path.join(args.log_dir,args.run_name)
    ckpt_path = os.path.join(args.ckpt_dir,args.run_name)
    sample_path = os.path.join(args.sample_dir,args.run_name)
    test_path = os.path.join(args.test_dir, args.run_name)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir) 
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    if args.gpu :
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    
    if args.seed is not None :
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        print('Using seed' + str(args.seed))
    
    if not torch.cuda.is_available():
        if args.gpu:
            print('Cuda is not available, please disable gpu')
            exit()
    

        
        
    if args.phase == 'train':
        train()
    elif args.phase == 'test' :
        test()




