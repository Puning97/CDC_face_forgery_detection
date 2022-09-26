import os
import math
import time
import argparse
import config
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import timm
from tricks.temperature_scaling import ModelWithTemperature
from tricks.label_smoothing import LSR
import numpy as np
from dual_dataset_v1 import celebdf_iav_train_loader,celebdf_iav_test_loader,ffpp_iav_train_loader,ffpp_iav_test_loader,\
                            ffpp_emotion_train_loader,ffpp_id_train_loader,ffpp_emotion_test_loader,ffpp_id_test_loader

#from motion_part.motionmodel_v2 import motion_model_1 as create_model
#from TD_model.optical_model import optical_model_50_pre_3 as create_model
from feature_transformer_BACKBONE import Feature_Trans_img as create_model
from teacher_v3 import id_teacher,moion_teacher
from utils import evaluate_img,train_one_epoch_img_distillation
import loggerset
from loggerset import print_hyperpar

os.environ['CUDA_VISIBLE_DEVICES']=config.gpus
def main(args):
    #logger
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    loggername =config.logpath+'/log/'+ rq+config.channel_name+'_'+config.lossname+config.modelname+'depth'+str(config.vit_depth)+'head'+str(config.vit_head)+'_distillation_01_img_batch'+str(config.batch_size)+'.log'
    logger=loggerset.get_logger(loggername)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model = create_model(pretrained=True)
    model=create_model()
    teacher_id=id_teacher()
    teacher_motion=moion_teacher()
    #labels =LSR(n_classes=2,eps=0.1)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        teacher_id= nn.DataParallel(teacher_id)
        teacher_motion= nn.DataParallel(teacher_motion)
        #labels = nn.DataParallel(labels)

    model.to(device)
    teacher_id.to(device)
    teacher_motion.to(device)
    #labels.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9,0.999),weight_decay=10E-8)
    #lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.step, gamma=0.8, last_epoch=-1)
    print_hyperpar(logger)
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch_img_distillation(model=model,
                                                                 teacher_id=teacher_id,
                                                                 teacher_motion=teacher_motion,
                                                optimizer=optimizer,
                                                data_loader=ffpp_iav_train_loader,
                                                device=device,
                                                epoch=epoch,
                                                log=logger)
        #scaled_model= ModelWithTemperature(model)
        #scheduler.step()
        #scaled_model.set_temperature(ffpp_iav_test_loader)
        # test
        #evaluate_img(model=model,data_loader=ffpp_iav_test_loader,device=device,epoch=epoch,log=logger)
        evaluate_img(model=model,data_loader=ffpp_iav_test_loader,device=device,epoch=epoch,log=logger)
        evaluate_img(model=model,data_loader=celebdf_iav_test_loader,device=device,epoch=epoch,log=logger)
        rq1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        torch.save(model.state_dict(), "./weights/"+rq1+"distillation_01_model-{}.pth".format(epoch))







if __name__ == '__main__':
    seed_num = 1026
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=config.classes)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--lr', type=float, default=config.learning_rate)
    parser.add_argument('--lrf', type=float, default=config.lrf)
    parser.add_argument('--step', type=int, default=config.step)  # 每隔多少步学习率递减
        # 数据集所在根目录
        # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--efficient_net', type=int, default=config.efficient_net,
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--model-name', default='', help='create model name')

        # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                            help='initial weights path')
        # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    opt = parser.parse_args()

    main(opt)