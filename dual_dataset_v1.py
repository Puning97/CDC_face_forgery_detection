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

class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, loader=default_loader,loader2=gray_loader,mode='train'):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是label
        self.imgs = imgs
        #self.transform1 = bigtransform
        #self.transform2 = smalltransform
        #self.target_transform = target_transform
        self.loader = loader
        self.loader2 = loader2
        self.mode=mode
        self.imgloader = loader
    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img=self.imgloader(fn)
        img=np.asarray(img)
        #img,buffer=self.load_frames_new(fn,self.sumframes)
        if self.mode == 'train':
            transform = self.create_train_transforms(config.height)
              # 数据标签转换为Tensor
        else:
            transform = self.create_val_transform(config.height)
        img = transform(image=img)['image']
        img=img.transpose((2,0,1))

        #buffer = buffer.transpose((3, 0, 1, 2))
        return torch.tensor(img).float(),label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
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
'''
    def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            # GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
'''


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

