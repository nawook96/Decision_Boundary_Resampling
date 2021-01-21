import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np

import copy
import random

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split

from collections import OrderedDict

from model.vgg import VGG19, VGG19_feature, VGG19_db_train

from utils.store_result import save_feature_info, save_intermediate_feature_train, save_intermediate_feature_val
from utils.folder import ImageFolder
from utils.feature_modify import decision_boundary_resampling

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 2

def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG19 on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--experience-num', dest='num_exp', type=int, default=1,
                        help='order number of experiences')
    parser.add_argument('-a', '--add-percent', dest='add_percent', type=int, default=20,
                        help='percent of minor class samples')
    parser.add_argument('-l', '--feature-learning-rate', dest='learning_rate_feature', type=float, default=0.00005,
                        help='learning rate on feature training')
    parser.add_argument('-e', '--feature-epoch', dest='feature_epoch_iter', type=int, default=10,
                        help='epochs on feature training')
    parser.add_argument('-f', '--feature-add', dest='how_many_times_add_feature', type=int, default=18,
                        help='how many times add feature')
    return parser.parse_args()

def main(net,
        feature_net,
        dbt,
        args,
        device,
        dict_path):

    num_exp = args.num_exp

    # feature train param
    add_percent = args.add_percent
    feature_epoch_iter = args.feature_epoch_iter
    how_many_times_add_feature = args.how_many_times_add_feature
    learning_rate_feature = args.learning_rate_feature
    feature_modified_epoch_iter = []
    for i in range(1 ,how_many_times_add_feature + 1):
        feature_modified_epoch_iter.append(i * feature_epoch_iter)
    total_feature_epoch = feature_epoch_iter * (how_many_times_add_feature + 1)

    # data set path
    data_path = '../data/ulcer_detect'

    #image pre-processing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    #load image set
    trainSet = ImageFolder(root=data_path, transform=transform, \
    train='train', split_txt=f'./result/{num_exp}/data/')
    valSet = ImageFolder(root=data_path, transform=transform, \
    train='val', split_txt=f'./result/{num_exp}/data/')

    n_train = len(trainSet)
    n_val = len(valSet)

    for_extract = DataLoader(
        dataset=trainSet,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=valSet,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    global_step = 0

    # logging
    logging.info(f'''Starting training:
    Experience number:      {num_exp}
    Add persent:            {add_percent}
    Feature epoch:          {feature_epoch_iter}
    Times add feature:      {how_many_times_add_feature}
    Feature learning rate:  {learning_rate_feature}
    Device:                 {device.type}
    ''')

    save_feature_info(num_exp, add_percent,\
    feature_epoch_iter, how_many_times_add_feature, learning_rate_feature, device.type)

    # init optim & loss func
    optimizer_feature = torch.optim.Adam(dbt.parameters(),lr = learning_rate_feature)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1, last_epoch=-1)
    criterion = nn.BCELoss().to(device)

    # validation stage에서 best loss 찾아내고 해당 weight들 저장하기 위해
    best_dict = OrderedDict()
    best_val_loss = 999999999.0

    # feature extract
    normal_feature, ulcer_feature = store_feature(feature_net,device,for_extract,n_train)

    # feature들 저장
    trainImage = copy.copy(normal_feature)
    trainImage.extend(ulcer_feature)

    n_train = len(trainImage)

    trainTarget = []

    # 각 feature들에 대한 labeling
    for i in range(0, len(normal_feature)):
        trainTarget.append(np.array([1,0],dtype=np.float64))
    for i in range(0, len(ulcer_feature)):
        trainTarget.append(np.array([0,1],dtype=np.float64))
    
    # feature들과 label 한번에 묶기
    trainSet = list(zip(trainImage, trainTarget))

    for epoch in range(total_feature_epoch):
        # 이전과 똑같은 부분에서 feature modify
        if epoch != 0 and epoch%int(total_feature_epoch/(how_many_times_add_feature+1)) == 0:
            ulcer_feature, new_ulcer_len = decision_boundary_resampling(normal_feature,ulcer_feature,dbt,device,add_percent)
            trainImage = copy.copy(normal_feature)
            trainImage.extend(ulcer_feature)
            n_train = len(trainImage)

            trainTarget = []

            for i in range(0, len(normal_feature)):
                trainTarget.append(np.array([1,0],dtype=np.float64))
            for i in range(0, len(ulcer_feature)):
                trainTarget.append(np.array([0,1],dtype=np.float64))
            
            # 새롭게 trainset 정의
            trainSet = list(zip(trainImage, trainTarget))

            save_intermediate_feature_train(epoch, num_exp, 0,
                        0, True, len(ulcer_feature))

        # 2000 -> 2 classification 할 때 마다 해당 부분 weight를 기존 VGG19 Dual Net에 update
        up_dict = update_dict(net, dbt)
        net.load_state_dict(up_dict)
        # validation part
        val_loss = val_net(net,device,val_loader,criterion,n_val,epoch,num_exp)

        # resampling 끝난 이후 validation part에서 optimal한 loss 찾기
        if best_val_loss > val_loss and epoch > (total_feature_epoch-feature_epoch_iter-1):
            best_val_loss = val_loss
            # 찾으면 저장
            for key, val in net.state_dict().items():
                best_dict[key] = net.state_dict()[key]

        # data set shuffle
        random.shuffle(trainSet)
        # 2000 features -> 2 class classification 학습
        train_net(dbt, device, trainSet, optimizer_feature, criterion, n_train, epoch, \
                total_feature_epoch,num_exp)
    net.load_state_dict(best_dict)
    torch.save(net.state_dict(), f'./result/{num_exp}/state_dict/fm_model.pth')
        

def store_feature(net,device,for_extract,n_train):
    net.eval()
    normal_feature = []
    ulcer_feature = []
    with tqdm(total=n_train, desc=f'Feature Extract', unit='img') as pbar:
        for i, (imgs,target,_) in enumerate(for_extract):
            imgs = imgs.to(device=device)
            target = target.float()
            target = target.to(device=device)

            feature = net(imgs)
            feature = feature.detach().cpu()

            if target[0][0].item() == 1.0:
                normal_feature.extend(feature)
            else:
                ulcer_feature.extend(feature)

            pbar.update(imgs.shape[0])
    return normal_feature, ulcer_feature

def train_net(net,device,train_loader,optimizer,criterion,n_train,epoch,epochs,num_exp):
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for i, (features,target) in enumerate(train_loader):
            features = features.to(device=device)
            target = torch.Tensor(target)
            #target = target.float()
            target = target.to(device=device)

            output = net(features)

            output = output.unsqueeze(dim=0)
            target = target.unsqueeze(dim=0)

            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            _, target = torch.max(target.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (train)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
    accuracy = 100 * correct/total
    print(f'Accuracy train on the images : {accuracy}')
    epoch_loss = epoch_loss / n_train
    print(f'Train loss in epoch {epoch+1}, loss : {epoch_loss}')
    save_intermediate_feature_train(epoch, num_exp, accuracy,
                    epoch_loss, False, 0)

def val_net(net,device,val_loader,criterion,n_val,epoch, num_exp):
    net.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for i, (imgs, target,_) in enumerate(val_loader):
            imgs = imgs.to(device=device)
            target = target.float()
            target = target.to(device=device)

            output = net(imgs)

            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            _, target = torch.max(target.data, 1)
            #predicted = (output.data > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            epoch_loss += loss.item()

            for item in predicted==target:
                temp_target = target.cpu().numpy()
                if item.item() == 0:
                    if temp_target == 0:
                        path = 'FP'
                        FP = FP + 1
                    if temp_target == 1:
                        path = 'FN'
                        FN = FN + 1
                elif item.item() == 1:
                    if temp_target == 1:
                        path = 'TP'
                        TP = TP + 1
                    if temp_target == 0:
                        path = 'TN'
                        TN = TN + 1

            pbar.set_postfix(**{'loss (val)': loss.item()})
            pbar.update(imgs.shape[0])

        if (TP + FP) == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (FP + TN)

        accuracy = 100 * correct/total
        epoch_loss = epoch_loss / n_val
    print(f'Accuracy validation on the images : {accuracy}')
    print(f'Validation loss in epoch {epoch+1}, loss : {epoch_loss}')
    print(f'TP : {TP}, TN : {TN}, FP : {FP}, FN : {FN}')
    print(f'Precision : {precision}, Recall : {recall}, Specificity : {specificity}')
    save_intermediate_feature_val(epoch+1, num_exp, accuracy, epoch_loss, precision, recall, specificity,\
                        TP, TN, FP, FN)
    return epoch_loss

def update_dict(net, dbt):
    temp_dict = OrderedDict()
    for key, val in net.state_dict().items():
        temp_dict[key] = net.state_dict()[key]

    temp_dict['classifier.weight'] = dbt.state_dict()['classifier.weight']
    temp_dict['classifier.bias'] = dbt.state_dict()['classifier.bias']

    return temp_dict

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    dict_path = f'./result/{args.num_exp}/state_dict/model.pth'

    # 전체 VGG model
    net = VGG19(n_classes)
    net.to(device=device)
    net.load_state_dict(
        torch.load(dict_path, map_location=device)
    )

    # feature extract 하기 위한 model
    feature_net = VGG19_feature(n_classes)
    feature_net.to(device=device)
    feature_net.load_state_dict(
        torch.load(dict_path, map_location=device)
    )

    # 2000 features -> 2 classes 하는 부분만
    dbt_dict = OrderedDict()
    dbt_dict['classifier.weight'] = feature_net.state_dict()['classifier.weight']
    dbt_dict['classifier.bias'] = feature_net.state_dict()['classifier.bias']

    decision_boundary_train = VGG19_db_train(n_classes)
    decision_boundary_train.to(device=device)
    decision_boundary_train.load_state_dict(dbt_dict)
    # decision_boundary_train.load_state_dict(
    #     torch.load(dict_path, map_location=device)
    # )

    main(net=net,
        feature_net=feature_net,
        dbt=decision_boundary_train,
        args=args,
        device=device,
        dict_path=dict_path
    )
    del net
    del decision_boundary_train