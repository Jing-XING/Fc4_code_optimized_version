from __future__ import print_function
import os
import sys
aux_path = os.path.abspath('../auxiliary')
sys.path.append(aux_path)

import argparse
import random
import json
import time
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


from model_baseline import CreateNet, squeezenet1_1
from dataset import *
from Utils import *

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=4000, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--lrate', type=float, default=0.0003, help='learning rate')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--foldnum', type=int, default=0, help='fold number')
parser.add_argument('--model_name',type=str,default='baseline',help='name of the model to be train')
parser.add_argument('--cudanum',type=str,default='0',help='cuda number')

opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cudanum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


now = datetime.datetime.now()

model_path = '../saved_model/'+opt.model_name
if not os.path.exists(model_path):
    os.mkdir(model_path)
model_path = os.path.join(model_path,'fold'+str(opt.foldnum)+'.pth')

log_path = '../log'
log_path = os.path.join(log_path,opt.model_name)
if not os.path.exists(log_path):
    os.mkdir(log_path)

logtxt = os.path.join(log_path, 'fold' + str(opt.foldnum) + '.txt')

# visualization
writer=SummaryWriter(log_path)

train_loss = AverageMeter()
val_loss = AverageMeter()
# load data
dataset_train = ColorChecker(train=True, folds_num=opt.foldnum)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.workers)

len_dataset_train = len(dataset_train)
print('len_dataset_train:', len(dataset_train))
dataset_test = ColorChecker(train=False, folds_num=opt.foldnum)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=opt.workers)
len_dataset_test = len(dataset_test)
print('len_dataset_test:', len(dataset_test))
print('training fold %d' % opt.foldnum)

# create network
SqueezeNet = squeezenet1_1(pretrained=True)
network = CreateNet(SqueezeNet)
network.to(device)
# network = nn.DataParallel(network).to(device)

if opt.pth_path != '':
    print('loading pretrained model')
    network.load_state_dict(torch.load(opt.pth_path))
print(network)
with open(logtxt, 'a') as f:
    f.write(str(network) + '\n')

# optimizer
lrate = opt.lrate
optimizer = optim.Adam(network.parameters(), lr=lrate)
criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
# train
print('start train.....')
best_val_loss = 100.0
for epoch in range(opt.nepoch):
    # train mode
    time_use1 = 0
    train_loss.reset()
    network.train()
    start = time.time()
    for i, data in enumerate(dataloader_train):
        optimizer.zero_grad()
        img, label, fn = data
        img = img.cuda()
        label = label.cuda()
        pred = network(img)
        pred_ill = torch.nn.functional.normalize(torch.sum(torch.sum(pred, 2), 2), dim=1)
        loss = get_angular_loss(pred_ill, label)
        loss.backward()
        train_loss.update(loss.item())
        optimizer.step()
    time_use1 = time.time() - start
    writer.add_scalar('loss/train_loss', train_loss.avg, epoch)


        # val mode
    time_use2 = 0
    val_loss.reset()
    with torch.no_grad():
        if epoch % 5 == 0:
            val_loss.reset()
            network.eval()
            start = time.time()
            errors = []
            for i, data in enumerate(dataloader_test):
                img, label, fn = data
                img = img.cuda()
                label = label.cuda()
                pred = network(img)
                pred_ill = torch.nn.functional.normalize(torch.sum(torch.sum(pred, 2), 2), dim=1)
                loss = get_angular_loss(pred_ill, label)
                val_loss.update(loss.item())
                errors.append(loss.item())
            time_use2 = time.time() - start
            writer.add_scalar('loss/val_loss', val_loss.avg, epoch)

    mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
    try:
        print('Epoch: %d,  Train_loss: %f,  Val_loss: %f: T_Time: %f, V_time: %f' % (
        epoch, train_loss.avg, val_loss.avg, time_use1, time_use2))
    except:
        print('IOError...')
    if (val_loss.avg > 0 and val_loss.avg < best_val_loss):
        best_val_loss = val_loss.avg
        best_mean = mean
        best_median = median
        best_trimean = trimean
        best_bst25 = bst25
        best_wst25 = wst25
        best_pct95 = pct95
        torch.save(network.state_dict(), model_path)
    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "best_val_loss": best_val_loss,
        "mean": best_mean,
        "median": best_median,
        "trimean": best_trimean,
        "bst25": best_bst25,
        "wst25": best_wst25,
        "pct95": best_pct95
    }
    with open(logtxt, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
