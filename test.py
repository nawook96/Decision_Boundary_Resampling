import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from model.vgg import VGG19

from sklearn import metrics
import skimage.transform as st

from utils.folder import ImageFolder
from utils.store_result import save_test_info, save_intermediate_test

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 2

def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG19 on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--experience-num', dest='num_exp', type=int, default=1,
                        help='order number of experiences')
    parser.add_argument('-f', '--after-feature-modified', dest='after_feature_modified', action='store_true', default=False,
                        help='Load after feature modified model')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args,
        num,
        net,
        device):
    num = num
    
    batch_size = 1

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    data_path = '../data/ulcer_detect'

    testSet = ImageFolder(root=data_path, transform=transform, train='test', \
    split_txt=f'./result/{num}/data/')

    n_test = len(testSet)

    test_loader = DataLoader(
        dataset=testSet,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    global_step = 0

    logging.info(f'''Starting test:
    Test size:          {n_test}
    Feature modified:   {args.after_feature_modified} 
    Device:             {device.type}
    ''')

    save_test_info(num, n_test)

    criterion = nn.BCEWithLogitsLoss()

    optim_epoch = 0
    max_acc = 0.0
    min_loss = 9999999999.9999

    dic_path = ''
    if args.after_feature_modified:
        dic_path = 'after_resampling'
    else:
        dic_path = 'before_resampling'

    try:
        os.mkdir(f"./result/{num}/{dic_path}")
        os.mkdir(f"./result/{num}/{dic_path}/FN")
        os.mkdir(f"./result/{num}/{dic_path}/FP")
        os.mkdir(f"./result/{num}/{dic_path}/TN")
        os.mkdir(f"./result/{num}/{dic_path}/TP")
    except OSError:
        pass
    if args.after_feature_modified :
        net.load_state_dict(torch.load(f"./result/{num}/state_dict/fm_model.pth", map_location=device))
    else:
        net.load_state_dict(torch.load(f"./result/{num}/state_dict/model.pth", map_location=device))
    y_true, y_score = test_net(num,net,device,test_loader,criterion,n_test,dic_path,args.after_feature_modified)
    #PR curve, ROC curve 미구현
    #print(y_true.argmax(axis=1), y_true.argmax(axis=1).shape)
    #print(y_score.argmax(axis=1), y_score.argmax(axis=1).shape)
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true.argmax(axis=1), y_score=y_score.argmax(axis=1), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Ulcer')
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(f"./result/{num}/{dic_path}/ROC_Curve.png"), bbox_inches='tight')
    plt.close()

    p, r, _= metrics.precision_recall_curve(y_true=y_true.argmax(axis=1), probas_pred=y_score.argmax(axis=1), pos_label=1)
    pr_auc = metrics.auc(r,p)

    plt.figure()
    lw = 2
    plt.plot(r, p, color='darkorange',
            lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve for Ulcer')
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(f"./result/{num}/{dic_path}/PR_Curve.png".format(num)), bbox_inches='tight')
    plt.close()

def test_net(num, net, device, test_loader, criterion, n_test, dic_path, is_fm):
    net.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    labels = ['normal','ulcer']

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    #y_true = []
    y_true = np.empty((1, 2), dtype=np.float64)
    #y_score = []
    y_score = np.empty((1, 2), dtype=np.float64)

    params = list(net.parameters())

    with tqdm(total=n_test, desc='Test round', unit='img', leave=False) as pbar:
        for i, (imgs, target, filename) in enumerate(test_loader):
            imgs = imgs.to(device=device)
            target = target.float()
            target = target.to(device=device)

            output = net(imgs)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)

            y_true_temp = np.array([[target.cpu().detach().numpy()[0][0], target.cpu().detach().numpy()[0][1]]])
            y_score_temp = np.array([[output.cpu().detach().numpy()[0][0], output.cpu().detach().numpy()[0][1]]])

            y_true = np.append(y_true, y_true_temp, axis=0)
            y_score = np.append(y_score, np.exp(y_score_temp), axis=0)

            _, target = torch.max(target.data, 1)
            #predicted = (output.data > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            epoch_loss += loss.item()
            path = ''
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
                vutils.save_image(imgs, f"./result/{num}/{dic_path}/{path}/{filename}.JPG")
            pbar.set_postfix(**{'loss (test)': loss.item()})
            pbar.update(imgs.shape[0])

        if (TP + FP) == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (FP + TN)

        accuracy = 100 * (correct/total)
        epoch_loss = epoch_loss / n_test

    print(f'Accuracy test on the images : {accuracy}')
    print(f'Test loss : {epoch_loss}')
    print(f'Precision : {precision}, Recall : {recall}, Specificity : {specificity}')
    save_intermediate_test(num, accuracy, epoch_loss, precision, recall, specificity,\
                        TP, TN, FP, FN, dic_path, is_fm)
    y_true = np.delete(y_true, [0, 0], axis=0)
    y_score = np.delete(y_score, [0, 0], axis=0)
    return y_true, y_score

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    net = VGG19(n_classes)

    net.to(device=device)

    n = args.num_exp

    main(args=args,
        num=n,
        net=net,
        device=device)
    del net