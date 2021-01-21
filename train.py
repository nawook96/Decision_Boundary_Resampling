import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.vgg import VGG19

from utils.split import no_people_split, ul_people_split, no_instance_split, ul_instance_split
from utils.store_result import save_train_info, save_intermediate_train, save_intermediate_val
from utils.folder import ImageFolder

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 2

def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG19 on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=25,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=20,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0000001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-s', '--seperate', dest='seperate', type=str, default='instance',
                        help='you can choose people, instance, fixed_data')
    parser.add_argument('-t', '--train_type', dest='train', type=str, default='TwoEncoder',
                        help='you can choose TwoEncoder, Mynpy')
    parser.add_argument('-n', '--experience-num', dest='num_exp', type=int, default=1,
                        help='order number of experiences')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(net,
        args,
        device,
        dir_checkpoint='./checkpoints'):

    num_exp = args.num_exp

    # data & init param
    seperate_by = args.seperate
    train_type = args.train

    # train param
    epochs = args.epochs
    batch_size = args.batchsize
    learning_rate = args.learning_rate

    # data set path
    if seperate_by == 'people' or seperate_by == 'instance':
        split_data(num_exp, seperate_by)

    # data set path
    data_path = '../data/ulcer_detect'

    # image pre-processing
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

    train_loader = DataLoader(
        dataset=trainSet,
        batch_size=batch_size,
        shuffle=True,
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

    # logging
    logging.info(f'''Starting training:
    Experience number:      {num_exp}
    Seperate type:          {seperate_by}
    Train type:             {train_type}
    Epoch:                  {epochs}
    Batch size:             {batch_size}
    Learning rate:          {learning_rate}
    Training size:          {n_train}
    Validation size:        {n_val}
    Device:                 {device.type}
    ''')

    save_train_info(num_exp, seperate_by, train_type, epochs, batch_size, learning_rate,\
                    n_train, n_val, device.type)

    # init optim & loss func
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1, last_epoch=-1)
    criterion = nn.BCELoss().to(device)
    #criterion = nn.CrossEntropyLoss().to(device)

    best_val_epoch = 0
    best_val_loss = 999999999.0

    loss_flow = []

    for epoch in range(epochs):
        # train part
        train_loss = train_net(net,device,train_loader,optimizer,criterion,n_train,epoch,epochs,num_exp)
        # validation part
        val_loss = val_net(net,device,val_loader,criterion,n_val,epoch,num_exp)
        # check best validation set loss on each epoch
        loss_flow.append(train_loss)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.state_dict(),
            dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

    plt.plot(range(0, epochs), loss_flow, color='blue',
            lw=2, label='loss for each epoch')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss for each epoch')
    plt.savefig(
        os.path.join(f"./result/{num_exp}/Train_Loss.png"), bbox_inches='tight')
    plt.close()
    # save best validation set loss model
    net.load_state_dict(
        torch.load(f'./checkpoints/{num_exp}/CP_epoch{best_epoch}.pth', map_location=device)
    )
    torch.save(net.state_dict(), f'./result/{num_exp}/state_dict/model.pth')
        

def train_net(net,device,train_loader,optimizer,criterion,n_train,epoch,epochs,num_exp):
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    count = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for i, (imgs,target,_) in enumerate(train_loader):
            imgs = imgs.to(device=device)
            target = target.float()
            target = target.to(device=device)

            output = net(imgs)
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

            count += 1

            pbar.update(imgs.shape[0])
        accuracy = 100 * correct/total
    print(f'Accuracy train on the images : {accuracy}')
    epoch_loss = epoch_loss / count
    print(f'Train loss in epoch {epoch+1}, loss : {epoch_loss}')
    save_intermediate_train(epoch+1, num_exp, accuracy, epoch_loss)
    return epoch_loss

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
    save_intermediate_val(epoch+1, num_exp, accuracy, epoch_loss, precision, recall, specificity,\
                            TP, TN, FP, FN)
    return epoch_loss

def split_data(num, seperate_by):
    try:
        os.mkdir(f"./result/{num}")
        os.mkdir(f"./result/{num}/state_dict")
        os.mkdir(f"./result/{num}/data")
    except OSError:
        pass
    if seperate_by == 'people':
        print('Seperated by people randomly..')
        no_people_split(src='../data/ulcer_detect/normal/',no=4500, num=num)
        ul_people_split(src='../data/ulcer_detect/ulcer/',ul=1000, num=num)
    elif seperate_by == 'instance':
        print('Seperated by instance randomly..')
        no_instance_split(src='../data/ulcer_detect/normal/',no=4500, num=num)
        ul_instance_split(src='../data/ulcer_detect/ulcer/',ul=1000, num=num)
    print('Finish')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    my_npy_path = ''

    train_type = args.train

    assert train_type == 'TwoEncoder' or train_type == 'Mynpy', 'you must set TwoEncoder or Mynpy'

    net = VGG19(n_classes)
    print('parameters of model : ', count_parameters(net))
    net.to(device=device)

    dir_checkpoint = f'checkpoints/{args.num_exp}/'

    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

    if train_type == 'Mynpy':
        net.load_state_dict(
            torch.load(my_npy_path, map_location=device)
        )
        logging.info(f'Model loaded from {my_npy_path}')

    try:
        main(net=net,
            args=args,
            device=device,
            dir_checkpoint=dir_checkpoint
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    del net