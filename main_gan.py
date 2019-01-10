from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import torch.backends.cudnn as cudnn
from dbpn_v1 import Net as DBPNLL
from dbpn import Net as DBPN
#from dbpn_iterative import Net as DBPNITER
from discriminator import Discriminator, FeatureExtractor, FeatureExtractorResnet
from data import get_training_set
from random import randrange
import pdb
import socket
import time
import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--pretrained_iter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=25, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--patch_size', type=int, default=60, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='dl00DBPNLLPIRM_pretrained_50.pth', help='sr pretrained base model')
parser.add_argument('--load_pretrained', type=bool, default=False)
parser.add_argument('--pretrained_D', default='dnnDBPNLLPIRM_RESNET_epoch_Discriminator_499.pth', help='sr pretrained base model')
parser.add_argument('--load_pretrained_D', type=bool, default=False)
parser.add_argument('--feature_extractor', default='VGG', help='Location to save checkpoint models')
parser.add_argument('--w1', type=float, default=1e-2, help='MSE weight')
parser.add_argument('--w2', type=float, default=1e-1, help='Perceptual weight')
parser.add_argument('--w3', type=float, default=1e-3, help='Adversarial weight')
parser.add_argument('--w4', type=float, default=10, help='Style weight')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='PIRM_VGG', help='Location to save checkpoint models')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cuda = opt.gpu_mode
cudnn.benchmark = True
print(opt)

def train_pretrained(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0], batch[1]
        minibatch = input.size()[0]
        for j in range(minibatch):
            input[j] = utils.norm(input[j],vgg=True)
            target[j] = utils.norm(target[j],vgg=True)
        
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])

        optimizer.zero_grad()
        sr = model(input)
        loss = MSE_loss(sr, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("Epoch: [%2d] [%4d/%4d] G_loss_pretrain: %.8f"
                          % ((epoch), (iteration), len(training_data_loader), loss.data))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def train(epoch):
    G_epoch_loss = 0
    D_epoch_loss = 0
    feat_epoch_loss = 0
    style_epoch_loss = 0
    adv_epoch_loss = 0
    mse_epoch_loss = 0
    model.train()
    D.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0], batch[1]
        minibatch = input.size()[0]
        real_label = torch.ones(minibatch) #torch.rand(minibatch,1)*0.5 + 0.7
        fake_label = torch.zeros(minibatch) #torch.rand(minibatch,1)*0.3
        
        for j in range(minibatch):
            input[j] = utils.norm(input[j],vgg=True)
            target[j] = utils.norm(target[j],vgg=True)
        
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            real_label = Variable(real_label).cuda(gpus_list[0])
            fake_label = Variable(fake_label).cuda(gpus_list[0])

        # Reset gradient
        D_optimizer.zero_grad()
        
        # Train discriminator with real data
        D_real_decision = D(target)
        D_real_loss = BCE_loss(D_real_decision, real_label)
        
        # Train discriminator with fake data
        recon_image = model(input)
        D_fake_decision = D(recon_image)
        D_fake_loss = BCE_loss(D_fake_decision, fake_label)
        
        D_loss = D_real_loss + D_fake_loss
        
        # Back propagation
        D_loss.backward()
        D_optimizer.step()
        
        # Reset gradient
        optimizer.zero_grad()
        
        # Train generator
        recon_image = model(input)
        D_fake_decision = D(recon_image)
        
        # Adversarial loss
        GAN_loss = opt.w3 * BCE_loss(D_fake_decision, real_label)
        
        # Content losses
        mse_loss = opt.w1 * MSE_loss(recon_image, target)
        
        #Perceptual loss
        x_VGG = Variable(batch[1].cuda())
        recon_VGG = Variable(recon_image.data.cuda())
        real_feature = feature_extractor(x_VGG)
        fake_feature = feature_extractor(recon_VGG)

        vgg_loss = opt.w2 * sum([ MSE_loss(fake_feature[i], real_feature[i].detach()) for i in range(len(real_feature))])        
        style_loss = opt.w4 * sum([ MSE_loss(utils.gram_matrix(fake_feature[i]), utils.gram_matrix(real_feature[i]).detach()) for i in range(len(real_feature))])

        # Back propagation
        G_loss = mse_loss + vgg_loss + GAN_loss + style_loss

        G_loss.backward()
        optimizer.step()
        
        # log
        G_epoch_loss += G_loss.data
        D_epoch_loss += D_loss.data
        feat_epoch_loss += (vgg_loss.data)
        style_epoch_loss += (style_loss.data)
        adv_epoch_loss += (GAN_loss.data)
        mse_epoch_loss += (mse_loss.data)
        print("Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f, mse:%.8f, perceptual: %.8f, style: %.8f, adv: %.8f"
                      % ((epoch), (iteration), len(training_data_loader), G_loss.data, D_loss.data,mse_loss.data, vgg_loss.data, style_loss.data, GAN_loss.data))

    print("===> Epoch {} Complete: Avg. Loss G: {:.4f} D: {:.4f} MSE: {:.4f} Perceptual: {:.4f} Style: {:.4f} Adv: {:.4f}".format(epoch, G_epoch_loss / len(training_data_loader), D_epoch_loss / len(training_data_loader), mse_epoch_loss/ len(training_data_loader), feat_epoch_loss/ len(training_data_loader),style_epoch_loss/ len(training_data_loader), adv_epoch_loss/ len(training_data_loader) ))

def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def checkpoint(epoch, pretrained_flag=False):
    if pretrained_flag:
        model_out_path = opt.save_folder+hostname+opt.model_type+opt.prefix+"_pretrained_{}.pth".format(epoch)
    else:
        model_out_path = opt.save_folder+hostname+opt.model_type+opt.prefix+opt.feature_extractor+"_epoch_{}.pth".format(epoch)
        model_out_path_D = opt.save_folder+hostname+opt.model_type+opt.prefix+opt.feature_extractor+"_epoch_Discriminator_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    torch.save(D.state_dict(), model_out_path_D)
    print("Checkpoint saved to {}".format(model_out_path))

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor)
#elif opt.model_type == 'DBPN-RES-MR64-3':
#    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) 
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 

    
model = torch.nn.DataParallel(model, device_ids=gpus_list)

###Discriminator
D = Discriminator(num_channels=3, base_filter=64, image_size=opt.patch_size*opt.upscale_factor)
D = torch.nn.DataParallel(D, device_ids=gpus_list)

###Feature Extractor
if opt.feature_extractor=='VGG':
    feature_extractor = FeatureExtractor(models.vgg19(pretrained=True))
else:
    feature_extractor = FeatureExtractorResnet(models.resnet152(pretrained=True))

###LOSS
MSE_loss = nn.MSELoss()
BCE_loss = nn.BCELoss()

print('---------- Generator architecture -------------')
utils.print_network(model)
print('---------- Discriminator architecture ---------')
utils.print_network(D)
print('-----------------------------------------------')

if opt.load_pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if opt.load_pretrained_D:
    D_name = os.path.join(opt.save_folder + opt.pretrained_D)
    if os.path.exists(D_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        D.load_state_dict(torch.load(D_name, map_location=lambda storage, loc: storage))
        print('Pre-trained Discriminator model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    D = D.cuda(gpus_list[0])
    feature_extractor = feature_extractor.cuda(gpus_list[0])
    MSE_loss = MSE_loss.cuda(gpus_list[0])
    BCE_loss = BCE_loss.cuda(gpus_list[0])
    
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

##PRETRAINED
if opt.pretrained:
    print('Pre-training starts.')
    for epoch in range(1, opt.pretrained_iter + 1):
        train_pretrained(epoch)
    print('Pre-training finished.')
    checkpoint(epoch, pretrained_flag=True)

###GAN Training
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        for param_group in D_optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('D: Learning rate decay: lr={}'.format(D_optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
