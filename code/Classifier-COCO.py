
import os 
import json
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch
from PIL import Image, ImageDraw
from torch.optim import lr_scheduler
import tqdm
import numpy as np
from collections import Counter
import math
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


import sys
sys.path.insert(1, './MSCOCO-IC')
import dataloader

ds_trans = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
])



def train_model(net, mscoco, criterior, optimizer, num_epochs, save_path, kind):

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.train()
    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True

    phase = 'train'

    batch_size = 100
    num_batches = 100 #int(math.ceil(mscoco.num_train/batch_size))
#     import pdb; pdb.set_trace()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        epoch_loss = 0.0
        epoch_corrects = 0


        for ibatch in range(num_batches):
            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, mscoco.num_train)



            a, b, y_batch =  mscoco.get_batch(bstart, bend, mode='train')
            x_batch = a if kind == 'box' else b

            x_batch = x_batch.astype('uint8')
            inputs = torch.zeros((x_batch.shape[0], x_batch.shape[3], 224,224))

            for idx,x in enumerate(x_batch):
             x = Image.fromarray(x)
             inputs[idx] = ds_trans(x)[None]

            labels = torch.tensor(y_batch)

            inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs.permute(0,3,1,2)
            optimizer.zero_grad()
            import pdb; pdb.set_trace()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                #labels = labels
                loss = criterior(outputs, labels)
                _, preds = torch.max(outputs, axis=1)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.shape[0]
                epoch_corrects += torch.sum(preds==labels.data)


        epoch_loss = epoch_loss / (num_batches )
        epoch_accuracy = epoch_corrects.double() /  (num_batches * batch_size)

        print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_path)




def test_model_clean(net, mscoco, kind):
    correct = 0
    total = 0

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)
    net.eval()
    
    
    batch_size = 100
    num_batches = 100 #int(math.ceil(mscoco.num_val/batch_size))


    for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, mscoco.num_val)


        a, b, y_batch =  mscoco.get_batch(bstart, bend, mode='val')
        x_batch = a if kind == 'box' else b

        x_batch = x_batch.astype('uint8')
        inputs = torch.zeros((x_batch.shape[0], x_batch.shape[3], 224,224))

        for idx,x in enumerate(x_batch):
         x = Image.fromarray(x)
         inputs[idx] = ds_trans(x)[None]

        labels = torch.tensor(y_batch)


        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
    #     correct += (predicted == labels.cuda()).sum()
        correct += (predicted == labels).sum()

    acc = float(correct) / total #(num_batches * batch_size)
        
    return acc#, images    



def main():

  mscoco = dataloader.DataLoader(data_dir = './MSCOCO-IC/')

  import sys



  kind = sys.argv[1]

  # train
  NUM_CLASSES = 10 
  resnet = models.resnet18(pretrained=True)
  num_ftrs = resnet.fc.in_features
  resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  #resnet.load_state_dict(torch.load('./coco_'+kind+'.pth'))
  train_model(resnet, mscoco, criterion, optimizer, 20, './coco_'+kind+'.pth', kind)


  # test
  # resnet.load_state_dict(torch.load('./coco_'+kind+'.pth'))
  # acc = test_model_clean(resnet, mscoco, kind)
  # print(f'accuracy is {acc}')



if __name__ == "__main__":
    main()
