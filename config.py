height=224
width=224
batchsize=32
import os
import torch

#GPU SET
#model
modelname='efficientnet_b3a'
#path
root='ff++_img'
logpath='/data/logpath'
ckpath=''#checkpoint path
#dataloader
imgsize=height

channel_name='RGB'#三通道的RGB，单通道的L
batch_size = batchsize #dataloader的batchsize
config_num_workers= 32 #dataloader的numworkers
#train
classes=2
epochs=10

learning_rate=0.0001
step=1
lrf=0.001
print_freq=300
eval_freq=5000
#loss function

distillation_alpha=0.85
label_smooothing_alpha=0.2
gamma=2
lossname='CELoss'
label_lamda=0.01
#audio
sr=40950
hop_length=35
#efficient_net
efficient_net=0

vit_head=12
vit_depth=2
depth=vit_depth
gpus='3,4,5'