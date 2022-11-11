import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2


def load_data(img_path, train=True):
    img_path = os.path.join('../../ds/dronebird', img_path)
    gt_path = os.path.join(os.path.dirname(img_path).replace('images', 'ground_truth'), 'GT_'+os.path.basename(img_path).replace('.jpg', '.h5'))
    # gt_path = img_path.replace('.jpg', '.h5').replace('data', 'annotation')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    if train:
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value < 0.25:
            dx = 0
            dy = 0
        elif rdn_value < 0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value < 0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)

        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        # target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        target = target[(int)(dy/2):(int)((crop_size[1]+dy)/2),
                        (int)(dx/2):(int)((crop_size[0]+dx)/2)]
        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    target = cv2.resize(
        target, (target.shape[1]//4, target.shape[0]//4), interpolation=cv2.INTER_CUBIC)*16
    return img, target
