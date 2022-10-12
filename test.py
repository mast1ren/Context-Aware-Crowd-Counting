from tabnanny import check
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error,mean_absolute_error

from torchvision import transforms
import json


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


with open('test.json','r') as f:
    img_paths=json.load(f)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

model = CANNet()
model = model.to(device)

checkpoint = torch.load('0model_best.pth.tar', map_location={'cuda:0': 'cuda:2'})
# checkpoint = torch.load('0checkpoint.pth.tar', map_location={'cuda:0': 'cuda:2'})
model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []
with torch.no_grad():
    for i in range(len(img_paths)):
        img = transform(Image.open(img_paths[i]).convert('RGB')).to(device)
        img = img.unsqueeze(0)
        h,w = img.shape[2:4]
        h_d = int(h/2)
        w_d = int(w/2)
        img_1 = Variable(img[:,:,:h_d,:w_d].to(device))
        img_2 = Variable(img[:,:,:h_d,w_d:].to(device))
        img_3 = Variable(img[:,:,h_d:,:w_d].to(device))
        img_4 = Variable(img[:,:,h_d:,w_d:].to(device))
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()
    
        # pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
        gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('data','annotation'),'r')
        groundtruth = np.asarray(gt_file['density'])
        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
        pred.append(pred_sum)
        gt.append(np.sum(groundtruth))
        print('\r[{:>{}}/{}], pred: {:.2f}, gt: {:.2f}'.format(i+1, len(str(len(img_paths))), len(img_paths), pred_sum, np.sum(groundtruth)), end='')
    print()
    
mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print ('MAE: ',mae)
print ('RMSE: ',rmse)