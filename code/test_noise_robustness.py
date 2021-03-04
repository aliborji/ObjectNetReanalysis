# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from model import *
from utils import detect_edge_batch, save_img
from dataset import folderDB
from make_imagenet_64_c import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--net_type', '-n', type=str,
                    choices=['rgb', 'rgbedge', 'edge'])

parser.add_argument('--data_dir', type=str)
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')

# Architecture
parser.add_argument('--model_name', '-m', type=str)
                    #choices=['tinyImgnet.pth'])
# Acceleration
# parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

# if args.model_name == 'tinyImgnet':
#     net = resnext_50_32x4d
#     net.load_state_dict(torch.load(''))
#     args.test_bs = 64


data_dir = args.data_dir # 'tiny-imagenet-200'
# inp_size = 64
# n_classes = 50 #200


net_type = args.net_type.lower()

net, _, _, _ = build_model_resNet(net_type, args.data_dir, args.inp_size, args.classes)
net.load_state_dict(torch.load(args.model_name,map_location=torch.device('cpu')))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)




args.test_bs = 64
args.prefetch = 4

# for p in net.parameters():
#     p.volatile = True

# if args.ngpu > 1:
#     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# if args.ngpu > 0:
#     net.cuda()

torch.manual_seed(1)
np.random.seed(1)
# if args.ngpu > 0:
#     torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded')

# /////////////// Data Loader ///////////////

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

mean = [0.3403, 0.3121, 0.3214]
std = [0.2724, 0.2608, 0.2669]

# clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
#     root="./tiny-imagenet-200/testset/images/",
#     # transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
#     transform=trn.Compose([trn.Resize(64), trn.ToTensor(), trn.Normalize(mean, std)])),    
#     batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []


    transform=trn.Compose([ trn.ToTensor(), trn.Normalize(mean, std)])
    # transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)])

    for severity in range(1,6):
        # distorted_dataset = dset.ImageFolder(
        #     # root='/share/data/vision-greg/DistortedImageNet/JPEG/' + distortion_name + '/' + str(severity),
        #     root='./Tiny-ImageNet-C/' + distortion_name + '/' + str(severity),                        
        #     # transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))
        #     transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]))            


        distorted_dataset = folderDB(
            root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor()]), net_type=net_type, base_folder=data_dir)
            # root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]), net_type=net_type, base_folder=data_dir)

        if net_type == 'edge':    
            distorted_dataset = folderDB(
                root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor()]), net_type='rgbedge', base_folder=data_dir)
                # root_dir='.', train=False,  transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]), net_type=net_type, base_folder=data_dir)



        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=2)



        # dataloader_dict = {'train': trainloader, 'val': testloader}


        # distorted_dataset = dset.ImageFolder(
        #     # root='/share/data/vision-greg/DistortedImageNet/JPEG/' + distortion_name + '/' + str(severity),
        #     root='./Tiny-ImageNet-C/' + distortion_name + '/' + str(severity),                        
        #     # transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))
        #     transform=trn.Compose([trn.CenterCrop(64), trn.ToTensor(), trn.Normalize(mean, std)]))            
        # distorted_dataset_loader = torch.utils.data.DataLoader(
        #     distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)



        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data, target = data.to(device), target.to(device)
            # data = V(data.cuda(), volatile=True)
            # import pdb; pdb.set_trace()

            if batch_idx>0: break
            for idx,d in enumerate(data):
                if idx>0: break
                # import pdb; pdb.set_trace()                
                # exr = distortion_name + '(d*255,severity)'
                exr = distortion_name + '(d[:3].permute(1,2,0)*255,severity)'
                aa = torch.Tensor(eval(exr))
                data[idx][:3] = transform((aa/255).numpy())
                # out_img = data[idx][:3].permute(1,2,0)

                # uncomment this to save some sample images
                # out_img = (out_img - out_img.min())/ (out_img.max() - out_img.min())
                # out_img = out_img*255
                # save_img(out_img, "sampleRobustness/{}".format(distortion_name+str(severity)+'.jpg'))
            # idx = 0    
            
            if net_type in ['rgbedge', 'edge']:
                # import pdb; pdb.set_trace()                
                # edge_maps = torch.zeros((data.shape[0],1,data.shape[2],data.shape[2]))
                # data = torch.cat((data, edge_maps),dim=1)#[None]
                data = detect_edge_batch(data)

                if net_type == 'edge':
                    data = data[:,3].unsqueeze(1)

                    # saving some sample edge maps
                    # import pdb; pdb.set_trace()                
                    # out_img = data[idx][:3].squeeze(0)
                    # out_img = (out_img - out_img.min())/ (out_img.max() - out_img.min())
                    # out_img = out_img*255
                    # save_img(out_img, "sampleRobustness/edge_{}".format(distortion_name+str(severity)+'.jpg'))


            # if net_type == 'rgbedge':
            #     import pdb; pdb.set_trace()                
            #     edge_maps = torch.zeros((data.shape[0],1,data.shape[2],data.shape[2]))
            #     data = torch.cat((data, edge_maps),dim=1)#[None]
            #     data = detect_edge_batch(data)

            output = net(data)

            pred = output.data.max(1)[1]
            # correct += pred.eq(target.cuda()).sum()
            correct += pred.eq(target).sum()
 

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', tuple(errs))
    return np.mean(errs)


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 
    'motion_blur', 
    'zoom_blur',
    'snow', 
    'frost', 
    'fog', 
    'brightness',
    'contrast', 
    'elastic_transform', 
    'pixelate', 
    'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]



error_rates = []
for distortion_name in distortions:
    print(distortion_name)
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))

