#!/usr/bin/env python
# coding: utf-8

# ## Note: You may need to adjust some paths in the following code. I moved some stuff around last minute!

# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

from overlapped_classes import overlapped_classes
import json
from PIL import Image
import scipy
from make_imagenet_64_c import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[4]:


with open("../mappings/objectnet_to_imagenet_1k.json") as f:
    data = json.load(f)

with open("../mappings/pytorch_to_imagenet_2012_id.json") as f:
    idxMap = json.load(f)
        
with open("../mappings/folder_to_objectnet_label.json") as f:
    folderMap = json.load(f)

with open('imagenet_classes.txt') as f:
# with open('../mappings/imagenet_to_label_2012_v2.txt') as f:    
    labels = [line.strip() for line in f.readlines()]    


# In[5]:


# inverting the folderMap
dirMap = {}
for u, v in folderMap.items():
    dirMap[v]= u


# In[6]:


from torchvision import models
dir(models)

# models.googlenet(pretrained=True)] #,

models_ = [#models.googlenet(pretrained=True), models.alexnet(pretrained=True), 
           models.vgg19(pretrained=True), 
           models.resnet152(pretrained=True),
           models.inception_v3(pretrained=True)] 
           #models.mnasnet1_0(pretrained=True)] #, models.resnext101_32x8d(pretrained=True), models.densenet161(pretrained=True)]


# In[7]:



# model = models.alexnet(pretrained=True)
# model = models.resnet152(pretrained=True)
# model = models.inception_v3(pretrained=True)
# model = models.googlenet(pretrained=True)

# for model_name in models_:
#     model = eval(model_name)
#     model.eval()


# In[8]:


def predict_image(image_path, box, draw=False, kind='Box', distortion='blur', severity=1):
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transform = transforms.Compose([
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ])
    
    
    if kind == 'Crop':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
        ])
    
    
    img = Image.open(image_path).convert("RGB")

    if kind == 'Box':
        xl, yl, xr, yr = box
        img = img.crop((xl, yl, xr, yr))   #((left, top, right, bottom)) 
        img = img.resize((224, 224))
        

    if kind == 'Full':
        # xl, yl, xr, yr = box
        # img = img.crop((xl, yl, xr, yr))   #((left, top, right, bottom)) 
        img = img.resize((224, 224))


    # apply the distortion    
#     import pdb; pdb.set_trace()    
    exr = distortion + '(img,severity)'
    aa = torch.Tensor(eval(exr))
    img_t = transform((aa/255).numpy()) # range [0,1]
        
        
    # import pdb; pdb.set_trace()
#     img_t = transform(img).float()
    
    
    if draw:
        plt.imshow(img) #img_t.permute((2,1,0)) )
        plt.show()

    image_tensor = img_t.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Predict the class of the image
    # import pdb; pdb.set_trace()
    output = model(image_tensor)
    
    confs, indices = torch.sort(output.data, descending=True)
    confs[0] = confs[0]/sum(confs[0])
    return np.abs(confs[0][:5]), indices[0][:5]  # index


# In[ ]:





# In[9]:


# /////////////// Display Results ///////////////
import collections

print('\nUsing ObjectNet data')

distortions = [
    'gaussian_noise', 
    'shot_noise', 
    'impulse_noise',
    'defocus_blur', 'glass_blur', 
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




# In[10]:


# data.items()


# In[3]:


kind = 'Box'    
error_rates = dict()
for distortion in distortions:
    print(distortion + '\n')
    error_rates[distortion] = [] 
    error_rates_distortion = dict() 
    
    for severity in range(4,5): #range(1,4):
        print(str(severity) + '\n')        
        error_rates_distortion[severity] = []

        for model in models_:
            model_name = model._get_name() #eval(model_name)
            model.eval()

            print(model_name + '\n')

            res_top1 = []
            res_top5 = []

            save_file = os.path.join('./outputs_robust/', distortion + '_' + str(severity) + '_' + model_name + '.txt')

            for n, category in enumerate(data):
                # if n > 1: break    
                txtfile = os.path.join('../Annotations', dirMap[category] + '.txt')
                if not os.path.exists(txtfile):
                    continue

                with open(txtfile) as f:
                    boxes = [line.strip() for line in f.readlines()]    

                # writing to the file
                with open(save_file, 'a+') as ff:
                    ff.write(f"{category} ---------------------------------------------------------------- \n")


                count = [0, 0]
                lines = 0
                for im in boxes[0:10]:
                    # import pdb; pdb.set_trace()
                    ss = im.split(' ')
                    fName = ss[0]
                    if len(ss) > 1:
                        lines += 1
                        coords = (int(i) for i in ss[1:] if i)
                        confs, idx = predict_image(os.path.join('../images/' + dirMap[category] + '/', fName), coords, False, kind, distortion, severity)

                        # top 1
                        count[0] += 1 if labels[idx[0]] in data[category] else 0  

                        # top 5
                        flag = False
                        for i in idx[0:]:
                              flag = flag or (labels[i] in data[category])
                        count[1] += 1 if flag else 0

                        # writing to the file
                        with open(save_file, 'a') as ff:
                            ff.write(f"{fName} {int(idx[0])} {int(idx[1])} {int(idx[2])} {int(idx[3])} {int(idx[4])}           {confs[0]:.2f} {confs[1]:.2f} {confs[2]:.2f} {confs[3]:.2f} {confs[4]:.2f} \n")

                accs = np.array(count)*100/lines

                print(f"{n} -> {category}: top 1: {accs[0]:.2f}  -  top 5: {accs[1]:.2f}           [num imgs: {lines}]")  

                with open(save_file, 'a+') as ff:
                    ff.write(f"{n} -> {category}: top 1: {accs[0]:.2f}  -  top 5: {accs[1]:.2f}           [num imgs: {lines}] \n")  

                res_top1.append(accs[0])        
                res_top5.append(accs[1])
            
            # avg. over categories
            sum1_model = sum(res_top1)/len(res_top1)
            sum5_model = sum(res_top5)/len(res_top5)
            print(sum1_model)
            print(sum5_model)
        
        # after severity
        error_rates_distortion[severity].append((model_name, sum1_model, sum5_model))
            
    error_rates[distortion].append(error_rates_distortion)    
    print(error_rates[distortion])


# In[4]:


import pickle

# a = {'hello': 'world'}

with open('error_robust_box.pickle', 'wb') as handle:
    pickle.dump(error_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# print a == b


# In[32]:


# pip install wand


# print(sum(res_top1)/len(res_top1))
# print(sum(res_top5)/len(res_top5))

# In[5]:


# # eval(model)
# # int(idx[0])
# print(f"{1.911333:.2f}")
# # print(f'{val:.2f}')

# confs, idx = predict_image(os.path.join('../images/' + dirMap[category] + '/', fName), coords, False)
# # confs / sum(confs)


# In[6]:


# eval(model_name)


# In[7]:


# models_ = [models.googlenet(pretrained=True), models.alexnet(pretrained=True), models.vgg19(pretrained=True), 
#            models.resnet152(pretrained=True), models.inception_v3(pretrained=True), 
#            models.mnasnet(pretrained=True), models.resnext101_32x8d(pretrained=True)]


# In[8]:


# m = models.mnasnet1_0(pretrained=True)


# In[9]:


# # dir(models)
# models_[0]._get_name()

