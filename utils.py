import sys
from sklearn import metrics
import torch
from tqdm import tqdm
import config
from loggerset import print_log_print
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pdb
import time
from tricks.label_smoothing import labelsmoothing
from dual_dataset_v1 import ffpp_iav_test_loader,celebdf_iav_test_loader,ffpp_id_test_loader,ffpp_emotion_test_loader
import loggerset
import numpy as np
from torch import nn
import pandas as pd
def kldivloss(weights,output,target,batchsize):
    loss_f = nn.KLDivLoss(size_average=False, reduce=False)
    loss_1 = loss_f(output, target)
    lossnew = torch.zeros(batchsize, 2)
    for i in range(batchsize):
        for j in range(2):
            lossnew[i][j] = weights[i] * loss_1[i][j]
    return torch.mean(lossnew)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 自己和打乱的自己进行叠加
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
def train_one_epoch_img(model, optimizer, data_loader, device, epoch,log):
    model.train()
    #if configs.lossname=='Focal':
        #loss_function = FocalLoss(class_num=configs.classes,gamma=configs.gamma)
    #elif configs.lossname=='CrossEntropyLoss':
    loss_function = torch.nn.CrossEntropyLoss()
    #loss_function = FocalLoss(class_num=config.classes, gamma=config.gamma)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images1, labels = data
        sample_num += images1.shape[0]
        pred = model(images1.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        #data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               #accu_loss.item() / (step + 1),
                                                                               #accu_num.item() / sample_num)
        if (step + 1) % config.print_freq == 0:
            print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f , acc : %.2f%%' % \
                                  (epoch, config.epochs, step + 1, len(data_loader), accu_loss.item() / (step + 1),
                                   accu_num.item() / sample_num),log)
        if (step+1) % config.eval_freq == 0:
            evaluate_img(model=model,data_loader=ffpp_id_test_loader,device=device,epoch=epoch,log=log)
            evaluate_img(model=model,data_loader=ffpp_emotion_test_loader,device=device,epoch=epoch,log=log)
            evaluate_img(model=model,data_loader=celebdf_iav_test_loader,device=device,epoch=epoch,log=log)
            model.train()
            rq1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            torch.save(model.state_dict(), "./weights/"+rq1+"_"+str(step)+"_model-{}.pth".format(epoch))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_one_epoch_img_labelsmoothing(model,optimizer, data_loader, device, epoch,log):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    label_function = labelsmoothing().to(device)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images1, labels = data
        sample_num += images1.shape[0]
        pred = model(images1.to(device))
        loss_label=label_function(pred,labels.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss1 = loss_function(pred, labels.to(device))
        loss=loss1*config.label_smooothing_alpha+loss_label*(1-config.label_smooothing_alpha)
        loss.backward()
        accu_loss += loss.detach()
        if (step + 1) % config.print_freq == 0:
            print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f , acc : %.2f%%' % \
                                  (epoch, config.epochs, step + 1, len(data_loader), accu_loss.item() / (step + 1),
                                   accu_num.item() / sample_num),log)
        if (step+1) % config.eval_freq == 0:
            evaluate_img(model=model,data_loader=ffpp_id_test_loader,device=device,epoch=epoch,log=log)
            evaluate_img(model=model,data_loader=ffpp_emotion_test_loader,device=device,epoch=epoch,log=log)
            evaluate_img(model=model,data_loader=celebdf_iav_test_loader,device=device,epoch=epoch,log=log)
            rq1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            torch.save(model.state_dict(), "./weights/"+rq1+"_"+str(step)+"_model-{}.pth".format(epoch))
            model.train()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_one_epoch_img_distillation(model, teacher_id,teacher_motion,optimizer, data_loader, device, epoch,log):
    model.train()
    teacher_id.eval()
    teacher_motion.eval()
    #if configs.lossname=='Focal':
        #loss_function = FocalLoss(class_num=configs.classes,gamma=configs.gamma)
    #elif configs.lossname=='CrossEntropyLoss':
    KL_loss = torch.nn.KLDivLoss(size_average=False, reduce=False)
    CE_loss = torch.nn.CrossEntropyLoss()
    label_function = labelsmoothing().to(device)
    #loss_function = FocalLoss(class_num=config.classes, gamma=config.gamma)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images1, labels = data
        sample_num += images1.shape[0]
        pred = model(images1.to(device))
        loss_label=label_function(pred,labels.to(device))
        #teacher_idpre=teacher_id(images1.to(device))
        idweights,teacher_idpre=teacher_id(images1.to(device))
        #teacher_motionpre=teacher_motion(images1.to(device))
        motionweights,teacher_motionpre=teacher_motion(images1.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        #loss = CE_loss(pred, labels.to(device))*config.distillation_alpha+(KL_loss(pred,teacher_idpre)+KL_loss(pred,teacher_motionpre))*(1-config.distillation_alpha)
        loss = CE_loss(pred, labels.to(device))*config.distillation_alpha+(kldivloss(idweights,pred,teacher_idpre,images1.shape[0])+kldivloss(motionweights,pred,teacher_motionpre,images1.shape[0]))*(1-config.distillation_alpha)/config.batchsize+loss_label*(1-config.label_smooothing_alpha)
        #pdb.set_trace()
        loss.backward()
        accu_loss += loss.detach()

        #data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               #accu_loss.item() / (step + 1),
                                                                               #accu_num.item() / sample_num)
        if (step + 1) % config.print_freq == 0:
            print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f , acc : %.2f%%' % \
                                  (epoch, config.epochs, step + 1, len(data_loader), accu_loss.item() / (step + 1),
                                   accu_num.item() / sample_num),log)
        if (step+1) % config.eval_freq == 0:
            evaluate_img(model=model,data_loader=ffpp_iav_test_loader,device=device,epoch=epoch,log=log)
            evaluate_img(model=model,data_loader=celebdf_iav_test_loader,device=device,epoch=epoch,log=log)
            rq1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            torch.save(model.state_dict(), "./weights/"+rq1+"_"+str(step)+"_model-{}.pth".format(epoch))
            model.train()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_img(model, data_loader, device, epoch,log):
    # loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        count = 0
        Acc = 0
        # Precision = 0
        Auc = 0
        F1 = 0
        # Recall = 0
        loss = 0
        a_all = []
        pre_all = []
        label_all = []
        EER = 0
        for images1,labels in data_loader:
            count += 1
            # images = images.reshape(-1, sequence_length, input_size).to(device)
            outputs = model(images1.to(device))

            a, predicted = torch.max(outputs.data, 1)
            a = a.data.cpu().numpy()
            predicted = predicted.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            a_all.extend(a)
            pre_all.extend(predicted)
            label_all.extend(labels)
            Acc += metrics.accuracy_score(labels, predicted)

            # Precision += metrics.precision_score(labels, predicted)
            # Auc += metrics.roc_auc_score(labels, predicted)
            #F1 += metrics.f1_score(labels, predicted)
            # Recall += metrics.recall_score(labels, predicted)
            # loss += loss_function(predicted,labels)
        fpr, tpr, thresholds = metrics.roc_curve(label_all, pre_all, pos_label=1)
        EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(EER)
        #for i in range(fpr.size):
            #if fpr[i] == (1 - tpr[i]):
                #EER = fpr[i]
        rq2 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        test = pd.DataFrame(data=a_all)
        test.to_csv('/data/puningyang/'+rq2+'.csv',encoding='gbk')
        print_log_print('epoch:{}, Acc: {:.4f} , AUC: {:.4f} , EER: {:.4f}, F1_Score: {:.4f}'.format(epoch, Acc / count,
                                                                         metrics.roc_auc_score(label_all, pre_all),
                                                                         EER, metrics.f1_score(label_all,pre_all)),log)

    return Acc / count, Acc / count


@torch.no_grad()
def evaluate_id_motion(idmodel,motionmodel, data_loader, device, epoch,log):
    # loss_function = torch.nn.CrossEntropyLoss()
    idmodel.eval()
    motionmodel.eval()
    with torch.no_grad():
        count = 0
        Acc = 0
        # Precision = 0
        Auc = 0
        F1 = 0
        # Recall = 0
        loss = 0
        idpre_all = []
        motionpre_all = []
        label_all = []
        EER = 0
        for images1,labels in data_loader:
            count += 1
            # images = images.reshape(-1, sequence_length, input_size).to(device)
            idoutputs = idmodel(images1.to(device))
            motionoutputs = motionmodel(images1.to(device))

            _, idpredicted = torch.max(idoutputs.data, 1)
            _, motionpredicted = torch.max(motionoutputs.data, 1)

            idpredicted = idpredicted.data.cpu().numpy()
            motionpredicted = motionpredicted.data.cpu().numpy()
            #labels = labels.data.cpu().numpy()

            idpre_all.extend(idpredicted)
            motionpre_all.extend(motionpredicted)
            #label_all.extend(labels)
            Acc += metrics.accuracy_score(motionpredicted, idpredicted)

            # Precision += metrics.precision_score(labels, predicted)
            # Auc += metrics.roc_auc_score(labels, predicted)
            #F1 += metrics.f1_score(labels, predicted)
            # Recall += metrics.recall_score(labels, predicted)
            # loss += loss_function(predicted,labels)
        fpr, tpr, thresholds = metrics.roc_curve(motionpre_all, idpre_all, pos_label=1)
        EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(EER)
        #for i in range(fpr.size):
            #if fpr[i] == (1 - tpr[i]):
                #EER = fpr[i]
        print_log_print('epoch:{}, Acc: {:.4f} , AUC: {:.4f} , EER: {:.4f}, F1_Score: {:.4f}'.format(epoch, Acc / count,
                                                                         metrics.roc_auc_score(motionpre_all, idpre_all),
                                                                         EER, metrics.f1_score(motionpre_all,idpre_all)),log)

    return Acc / count, Acc / count


def train_one_epoch_img_mixup(model, optimizer, data_loader, device, epoch, log):
    model.train()
    # if configs.lossname=='Focal':
    # loss_function = FocalLoss(class_num=configs.classes,gamma=configs.gamma)
    # elif configs.lossname=='CrossEntropyLoss':
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(class_num=config.classes, gamma=config.gamma)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images1, labels = data
        sample_num += images1.shape[0]
        miximg,labela,labelb,lam=mixup_data(images1,labels,alpha=0.5,use_cuda=True)
        pred = model(miximg.to(device))
        #pred_classes = torch.max(pred, dim=1)[1]
        #accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss_func = mixup_criterion(labela.to(device), labelb.to(device),lam)
        loss=loss_func(loss_function,pred)
        loss.backward()
        accu_loss += loss.detach()

        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        # accu_loss.item() / (step + 1),
        # accu_num.item() / sample_num)
        if (step + 1) % config.print_freq == 0:
            print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f , acc : %.2f%%' % \
                            (epoch, config.epochs, step + 1, len(data_loader), accu_loss.item() / (step + 1),
                             accu_num.item() / sample_num), log)
        if (step + 1) % config.eval_freq == 0:
            evaluate_img(model=model, data_loader=ffpp_id_test_loader, device=device, epoch=epoch, log=log)
            evaluate_img(model=model, data_loader=ffpp_emotion_test_loader, device=device, epoch=epoch, log=log)
            evaluate_img(model=model, data_loader=celebdf_iav_test_loader, device=device, epoch=epoch, log=log)
            model.train()
            rq1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            torch.save(model.state_dict(), "./weights/"+rq1+"_"+str(step)+"_model-{}.pth".format(epoch))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
