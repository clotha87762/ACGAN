# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import math

import numpy as np 

class Generator( nn.modules ):
    def __init__(self , args ):
        
        self.initsize = args.imsize // 8
        
        self.embedding = nn.Embedding( args.num_class , args.dim_embed )
        self.fc1 = nn.Linear( args.dim_embed , self.initsize**2 * args.gfdim * 8 )
        
        if args.deconv:
            
            pad = math.ceil( (args.g_kernel - 2) / 2)
            outpad = -((args.g_kernel - 2) - (2*pad))
            
            self.conv1 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 8 , args.gfdim * 4 , args.g_kernel ,stride = 2 , padding = pad , out_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim * 4),
                    nn.LeakyReLU(inplace = True)
                    )
            self.conv2 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 4 , args.gfdim * 2 , args.g_kernel ,stride = 2 , padding = pad , out_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim * 4),
                    nn.LeakyReLU(inplace = True)
                    )
            self.conv3 = nn.Sequential(
                    nn.ConvTranspose2d(args.gfdim * 2 , args.gfdim  , args.g_kernel ,stride = 2 , padding = pad , out_padding = outpad ),
                    nn.BatchNorm2d(args.gfdim * 4),
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
                    nn.Upsample(scle_factor = 2 ),
                    nn.BatchNorm2d(args.gfdim * 4),
                    nn.LeakyReLU(inplace = True )
                    )
            self.conv2 = nn.Sequential(
                    nn.Conv2d(args.gfdim * 4 , args.gfdim*2 , 3  , stride = 1 , padding = 1 ),
                    nn.Upsample(scle_factor = 2 ),
                    nn.BatchNorm2d(args.gfdim * 2),
                    nn.LeakyReLU(inplace = True )
                    )
            self.conv3 = nn.Sequential(
                    nn.Conv2d(args.gfdim * 2 , args.gfdim , 3 , stride = 1 , padding = 1 ),
                    nn.Upsample(scle_factor = 2 ),
                    nn.BatchNorm2d(args.gfdim ) ,
                    nn.LeakyReLU(inplace = True )
                    )
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gfdim , args.out_dim , 4 , stride = 1  , padding = 1),
                    nn.Tanh()
                    )
            
        
    def forward(self , noise ,  class_index ):
        latent = self.embedding(class_index)
        x = torch.mul( latent , noise)
        x = self.fc1(x)
        x = x.view( args.batch , self.initsize , 1 , 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.conv4(x)
        
        return x
    

class Discriminator( nn.modules ):
    
    def __init(self, args ):
        self.conv1 = nn.Sequential(
                nn.Conv2d(args.out_dim, args.dfdim , 3 , 2 ,1 ),
                nn.utils.spectral_norm() if args.sn else nn.BatchNorm2d(args.dfdim ),
                nn.LeakyReLU(inplace= True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(args.out_dim, args.dfdim*2 , 3 , 2 , 1 ),
                nn.utils.spectral_norm() if args.sn else nn.BatchNorm2d(args.dfdim * 2 ),
                nn.LeakyReLU(inplace= True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(args.out_dim*2 , args.dfdim*4 , 3 , 2 , 1 ),
                nn.utils.spectral_norm() if args.sn else nn.BatchNorm2d(args.dfdim * 4  ),
                nn.LeakyReLU(inplace= True)
                )
        if args.sn:
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gfdim * 4 , args.out_dim * 8  , 4 , stride = 1  , padding = 1),
                    nn.utils.spectral_norm() ,
                    nn.Tanh()
                    )
        else:
            self.conv4 = nn.Sequential(
                    nn.ZeroPad2d((1,0,1,0)),
                    nn.Conv2d(args.gfdim * 4 , args.out_dim * 8  , 4 , stride = 1  , padding = 1),
                    nn.Tanh()
                    )
        
        self.dsize = args.imsize // (8)
        
        self.fc_gan = nn.Linear( args.out_dim* 8 * (self.dsize**2) , 1)
        self.fc_aux1 = nn.Linear( args.out_dim* 8 * (self.dsize**2) , 128)
        self.fc_aux2 = nn.Linear( 128 , args.num_class)
    
    def forward(self , _input ):
        x = self.conv1(_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view( args.batch , args.out_dim* 8 * (self.dsize**2) )
        gan_out = self.fc_gan(x)
        aux_out = self.fc_aux(x)
        return gan_out , aux_out