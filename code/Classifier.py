#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import json
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
import tqdm
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.nxutils import points_inside_poly
from make_imagenet_64_c import *


# In[3]:

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



readPath = '/Users/ali/Desktop/objectnet-1.0/images/'


# In[4]:


objsNames = {'chair':0, 'broom':1, 'candle':2, 'basket':3, 'banana':4, 'iron_for_clothes':5, 'plate':6, 'fan':7, 'toaster':8, 't-shirt':9}


# In[5]:


objs = list(objsNames.keys())

# with open('./models/train_data.npy', 'rb') as f:
#     train_data = np.load(f)

with open('./models/val_data.npy', 'rb') as f:
    val_data = np.load(f,allow_pickle=True)




def crop_mask(img, pts_lst):
    img_array = np.asarray(img)
#             print(img_array.shape)
#             width, height = img.size
    # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    # width = ?
    # height = ?

    pts = np.array(pts_lst)
    pts = pts.astype('int')
#             print(pts)

    # read image as RGB (without alpha)
#             img = Image.open("crop.jpg").convert("RGB")

    # convert to numpy (for convenience)
#             img_array = numpy.asarray(img)

    # create mask
    polygon =  [(u[0],u[1]) for u  in pts] #  [(40,203),(623,243),(691,177),(581,26),(482,42)]
#             print(polygon)
    # create new image ("1-bit pixels, black and white", (width, height), "default color")
    mask_img = Image.new('1', (img_array.shape[1], img_array.shape[0]), 0)

    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)

    # assemble new image (uint8: 0-255)
    new_img_array = np.ones(img_array.shape, dtype='uint8')

    # copy color values (RGB)
    new_img_array[:,:,:3] = img_array[:,:,:3]

    # filtering image by mask
    new_img_array[:,:,0] = new_img_array[:,:,0] * mask
    new_img_array[:,:,1] = new_img_array[:,:,1] * mask
    new_img_array[:,:,2] = new_img_array[:,:,2] * mask

    # back to Image from numpy
    img = Image.fromarray(new_img_array, "RGB")

    return img




import collections

print('\nUsing ObjectNet data')

distortions = [
    'gaussian_noise', 
    'shot_noise', 
    'impulse_noise',
    'defocus_blur', 
    'glass_blur', 
    'motion_blur', 
    'zoom_blur',
    'snow', 
    'frost', 
    'fog', 
    'brightness',
    'contrast', 
    'elastic_transform', 
    # 'pixelate', 
    'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]
transform = transforms.Compose([
#         transforms.CenterCrop(224),
    transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
])


# In[8]:


kind = 'seg'  
NUM_CLASSES = 10 
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
resnet.load_state_dict(torch.load('./models/'+kind+'.pth'))


# In[ ]:



device = 'cpu'
resnet.eval()
error_rates = dict()
for distortion in distortions:
  print(distortion + '\n')
  error_rates[distortion] = [] 
#     error_rates_distortion = dict() 
  
  for severity in range(1,4):
      print(str(severity))        
#         error_rates_distortion[severity] = []

      res_top1, res_top5 = [], []
      correct = 0
      total = 0

      # for n, category in enumerate(data):
      for img_path, polygon, label in val_data:    
          images = Image.open(img_path).convert('RGB')
#             import pdb; pdb.set_trace()
#             images, labels = images[0].to(device), labels.to(device)         
          #images = np.array(img)#.permute(1,2,0)
          if kind == 'box':
              pts = np.array(polygon)
              pts = pts.astype('int')
              min_x, max_x, min_y, max_y =  pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max()      
  #             print([min_x, max_x, min_y, max_y])
  #             print(img.width)
              images = images.crop([min_x, min_y, max_x, max_y])    
  
          if kind == 'seg':
              images = crop_mask(images, polygon)
              pts = np.array(polygon)
              pts = pts.astype('int')
              min_x, max_x, min_y, max_y =  pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max()      
  #             print([min_x, max_x, min_y, max_y])
  #             print(img.width)
              images = images.crop([min_x, min_y, max_x, max_y])                  
  

          images = images.resize((224, 224))
          
          exr = distortion + '(images,severity)'
          aa = torch.Tensor(eval(exr))
          images = transform((aa/255).numpy()) # range [0,1]
             
#             import pdb; pdb.set_trace()    
          outputs = resnet(images[None])
          _, predicted = torch.max(outputs.data, 1)

          total += 1#label.size(0)
      #     correct += (predicted == labels.cuda()).sum()
          correct += (predicted == label).sum()

      acc = float(correct) / total
                  
      # after severity
          
      error_rates[distortion].append(acc)    
  print(error_rates[distortion])

