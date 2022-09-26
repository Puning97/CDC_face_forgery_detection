from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import config
from config import channel_name, batch_size, config_num_workers
from transforms.albu import IsotropicResize
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
import cv2
import numpy as np
import os
import random
import pdb
#uadfv='/data1/share/UADFV/uadfv_'
ffpp='/data1/share/ffpp/'
celebdf='/data1/share/Celeb-DF/celeb_'
dfdc=''


def default_loader(path):
    return Image.open(path).convert('RGB')
def gray_loader(path):
    return Image.open(path).convert('L')

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

class MyDataset(Dataset):  
    def __init__(self, txt, loader=default_loader,loader2=gray_loader,mode='train'):  
        super(MyDataset, self).__init__()  
        fh = open(txt, 'r')  
        imgs = []
        for line in fh: 
            line = line.strip('\n')
            line = line.rstrip('\n') 
            words = line.split()  
            imgs.append((words[0], int(words[1]))) 
        self.imgs = imgs
        self.loader = loader
        self.loader2 = loader2
        self.mode=mode
        self.imgloader = loader
    def __getitem__(self, index):  
        fn, label = self.imgs[index]  
        img=self.imgloader(fn)
        img=np.asarray(img)
        
        if self.mode == 'train':
            transform = self.create_train_transforms(config.height)
              
        else:
            transform = self.create_val_transform(config.height)
        img = transform(image=img)['image']
        img=img.transpose((2,0,1))

        
        return torch.tensor(img).float(),label  

    def __len__(self):  
        return len(self.imgs)

    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    def create_train_transforms(self,size=224):
        return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                        GaussNoise(p=0.1),
                        GaussianBlur(blur_limit=3, p=0.05),
                        HorizontalFlip(),
                        IsotropicResize(max_side=size),
                        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
                        ToGray(p=0.2),
                        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                                         border_mode=cv2.BORDER_CONSTANT, p=0.5),
                        ]
                       )
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])



#UADFV
#uadfv_img_train_data = MyDataset(txt=uadfv + 'train.txt', mode='train')
#uadfv_img_test_data = MyDataset(txt=uadfv + 'test.txt', mode='test')
#weights = make_weights_for_balanced_classes(uadfv_img_train_data.imgs, config.classes)
#weights = torch.DoubleTensor(weights)
#print('Weights: ',weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#uadfv_iav_train_loader = DataLoader(dataset=uadfv_img_train_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers, sampler = sampler, pin_memory=True)
#uadfv_iav_test_loader = DataLoader(dataset=uadfv_img_test_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers,pin_memory=True)
#print('num_of_trainData:', len(uadfv_img_train_data))
#print('num_of_testData:', len(uadfv_img_test_data))
#print('UADFV Ready')


#Celeb-DF
celebdf_img_train_data = MyDataset(txt=celebdf + 'train.txt', mode='train')
celebdf_img_test_data = MyDataset(txt=celebdf + 'test.txt', mode='test')
weights = make_weights_for_balanced_classes(celebdf_img_train_data.imgs, config.classes)
weights = torch.DoubleTensor(weights)
#print('Weights: ',weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
celebdf_iav_train_loader = DataLoader(dataset=celebdf_img_train_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers, sampler = sampler, pin_memory=False)
celebdf_iav_test_loader = DataLoader(dataset=celebdf_img_test_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers,pin_memory=False)
#print('num_of_trainData:', len(celebdf_img_train_data))
#print('num_of_testData:', len(celebdf_img_test_data))
print('Celeb-DF Ready')


#FF++
ffpp_all_train_data = MyDataset(txt=ffpp + 'ffpp_train.txt', mode='train')
ffpp_all_test_data = MyDataset(txt=ffpp + 'ffpp_test.txt', mode='test')
weights = make_weights_for_balanced_classes(ffpp_all_train_data.imgs, config.classes)
weights = torch.DoubleTensor(weights)
#print('Weights: ',weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

ffpp_iav_train_loader = DataLoader(dataset=ffpp_all_train_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers, sampler = sampler, pin_memory=False)
ffpp_iav_test_loader = DataLoader(dataset=ffpp_all_test_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers,pin_memory=False)
#print('num_of_trainData:', len(ffpp_img_train_data))
#print('num_of_testData:', len(ffpp_img_test_data))
print('FF++ Ready')

ffpp_id_train_data = MyDataset(txt=ffpp + 'id_train.txt', mode='train')
ffpp_id_test_data = MyDataset(txt=ffpp + 'id_test.txt', mode='test')
weights = make_weights_for_balanced_classes(ffpp_id_train_data.imgs, config.classes)
weights = torch.DoubleTensor(weights)
#print('Weights: ',weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

ffpp_id_train_loader = DataLoader(dataset=ffpp_id_train_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers, sampler = sampler, pin_memory=False)
ffpp_id_test_loader = DataLoader(dataset=ffpp_id_test_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers,pin_memory=False)
#print('num_of_trainData:', len(ffpp_img_train_data))
#print('num_of_testData:', len(ffpp_img_test_data))
print('FF++ id Ready')

ffpp_emotion_train_data = MyDataset(txt=ffpp + 'emotion_train.txt', mode='train')
ffpp_emotion_test_data = MyDataset(txt=ffpp + 'emotion_test.txt', mode='test')
weights = make_weights_for_balanced_classes(ffpp_emotion_train_data.imgs, config.classes)
weights = torch.DoubleTensor(weights)
#print('Weights: ',weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

ffpp_emotion_train_loader = DataLoader(dataset=ffpp_emotion_train_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers, sampler = sampler, pin_memory=False)
ffpp_emotion_test_loader = DataLoader(dataset=ffpp_emotion_test_data, batch_size=batch_size, shuffle=False, num_workers=config_num_workers,pin_memory=False)
#print('num_of_trainData:', len(ffpp_img_train_data))
#print('num_of_testData:', len(ffpp_img_test_data))
print('FF++ motion Ready')

