# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.autograd as autograd
import math

import numpy as np 


def gradient_penalty( discriminator , real_sample , fake_sample ):
    
    batch = real_sample.shape[0]

    alpha = torch.from_numpy( np.random.random(batch,1) )
    interpolate = (alpha * real_sample)  +  ((1.0-alpha) * fake_sample)
    output = discriminator(interpolate)
    
    grad_output = torch.empty([batch,1]).fill_(1.0)
    grads = autograd.grad( output, interpolate, grad_output)
    
    grads = grads.view(batch,-1)
    penalty = (grads.norm(p=2 , dim = 1) - 1.0) ** 2
    penalty = torch.mean(penalty)
    return penalty

class Generator( nn.Module ):
    
    def __init__(self , args ):
        super(Generator, self).__init__()
        self.initsize = args.imsize // 8
        self.gfdim = args.gfdim
        
        self.embedding = nn.Embedding( args.num_class , args.dim_embed )
        self.fc1 = nn.Linear( args.dim_embed , (self.initsize**2) * args.gfdim * 8 )
        
        self.batchnorm_fc = nn.BatchNorm2d(args.gfdim * 8)
        
        if args.deconv:
            
            pad = math.ceil( (args.g_kernel - 2) / 2)
            outpad = -((args.g_kernel - 2) - (2*pad))
            
            self.conv1 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 8 , args.gfdim * 4 , args.g_kernel ,stride = 2 , padding = pad , output_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim * 4),
                    nn.LeakyReLU(inplace = True)
                    )
            self.conv2 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 4 , args.gfdim * 2 , args.g_kernel ,stride = 2 , padding = pad , output_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim * 2),
                    nn.LeakyReLU(inplace = True)
                    )
            self.conv3 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 2 , args.gfdim  , args.g_kernel ,stride = 2 , padding = pad , output_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim ),
                    nn.LeakyReLU(inplace = True)
                    )
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gfdim , args.out_dim , 4 , stride = 1  , padding = 1),
                    nn.Tanh()
                    )
            
        else:
            pad = ((args.g_kernel-1))/2
            self.conv1 = nn.Sequential(
                    nn.Conv2d(args.gfdim * 8 , args.gfdim*4 , 3 , stride = 1 , padding = 1 ),
                    nn.BatchNorm2d(args.gfdim * 4),
                    nn.LeakyReLU(inplace = True ),
                    nn.Upsample(scale_factor = 2 ),
                    )
            self.conv2 = nn.Sequential(
                    nn.Conv2d(args.gfdim * 4 , args.gfdim*2 , 3  , stride = 1 , padding = 1 ),
                    nn.BatchNorm2d(args.gfdim * 2),
                    nn.LeakyReLU(inplace = True ),
                     nn.Upsample(scale_factor = 2 ),
                    )
            self.conv3 = nn.Sequential(
                    nn.Conv2d(args.gfdim * 2 , args.gfdim , 3 , stride = 1 , padding = 1 ),
                    nn.BatchNorm2d(args.gfdim ) ,
                    nn.LeakyReLU(inplace = True ),
                    nn.Upsample(scale_factor = 2 ),
                    )
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gfdim , args.out_dim , 4 , stride = 1  , padding = 1),
                    nn.Tanh()
                    )
            
        
    def forward(self , noise ,  class_index ):
        #print(noise.shape)
        #print(class_index.shape)
        latent = self.embedding(class_index)
        #print(latent.shape)
        x = torch.mul( latent , noise)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = x.view( x.shape[0] , self.gfdim * 8  , self.initsize , self.initsize)
        x = self.batchnorm_fc(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.conv4(x)
        
        return x
    

class Discriminator( nn.Module ):
    
    def __init__(self, args ):
        
        super(Discriminator, self).__init__()
        

        if args.sn:
            self.conv1 = nn.Sequential(
                nn.utils.spectral_norm( nn.Conv2d(args.out_dim , args.dfdim , 3 , 2 , 1 )),
                nn.LeakyReLU(inplace= True)
                )
            self.conv2 = nn.Sequential(
                nn.utils.spectral_norm( nn.Conv2d(args.dfdim , args.dfdim*2 , 3 , 2 , 1 )),
                nn.LeakyReLU(inplace= True)
                )
            self.conv3 = nn.Sequential(
                nn.utils.spectral_norm( nn.Conv2d(args.dfdim*2 , args.dfdim*4 , 3 , 2 , 1 )),
                nn.LeakyReLU(inplace= True)
                )
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.utils.spectral_norm(nn.Conv2d(args.dfdim * 4 , args.dfdim * 8  , 4 , stride = 1  , padding = 1)),
                    nn.Tanh()
                    )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(args.out_dim, args.dfdim , 3 , 2 ,1 ),
                nn.BatchNorm2d(args.dfdim ),
                nn.LeakyReLU(inplace= True)
                )
            self.conv1 = nn.Sequential(
                nn.Conv2d(args.dfdim, args.dfdim*2 , 3 , 2 ,1 ),
                nn.BatchNorm2d(args.dfdim *2),
                nn.LeakyReLU(inplace= True)
                )
            self.conv1 = nn.Sequential(
                nn.Conv2d(args.dfdim*2, args.dfdim*4 , 3 , 2 ,1 ),
                nn.BatchNorm2d(args.dfdim *4 ),
                nn.LeakyReLU(inplace= True)
                )
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gdfdim * 4 , args.dfdim * 8  , 4 , stride = 1  , padding = 1),
                    nn.Tanh()
                    )
        
        self.dsize = args.imsize // (8)
        self.dfdim = args.dfdim
        
        self.fc_gan = nn.Linear( args.dfdim* 8 * (self.dsize**2) , 1)
        self.fc_aux1 = nn.Linear( args.dfdim* 8 * (self.dsize**2) , 128)
        self.fc_aux2 = nn.Linear( 128 , args.num_class)
        self.fc_aux = nn.Linear(args.dfdim* 8 * (self.dsize**2) , args.num_class)
        
        self.out_dim = args.out_dim
        
        self.soft_max = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , _input ):
        x = self.conv1(_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view( _input.shape[0] , self.dfdim * 8 * (self.dsize**2) )
        
        gan_out= self.sigmoid(self.fc_gan(x))
        
        
        #aux_temp = self.fc_aux1(x)
        #aux_out = self.soft_max(self.fc_aux2(aux_temp))
        
        aux_out = self.soft_max(self.fc_aux(x))
        return gan_out , aux_out