from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
import argparse
import torch
import csv
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import utils as vutils

sys.path.append('/shareData3/lab-xing.jing/project/C4/auxiliary/')
from model_baseline import squeezenet1_1,CreateNet
from dataset_predmap_fc4  import *
sys.path.append("/shareData3/lab-xing.jing/project/C4/train/uda_method/")
from dataset_nus_and_color_checker import ColorChecker
from utils import *
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
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--lrate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--pth_path0', default="/shareData3/lab-xing.jing/project/C4/train/uda_method/models/train_cc_val_nus/uda_tag_sum_not_all_tag_foldnum0Canon1DsMkIII/model.pth", type=str)
parser.add_argument('--pth_path1', default="/shareData3/lab-xing.jing/project/quasi-unsupervised-cc/pretrain_model/c4_pretrain_model/fold1.pth", type=str)
parser.add_argument('--pth_path2', default="/shareData3/lab-xing.jing/project/quasi-unsupervised-cc/pretrain_model/c4_pretrain_model/fold2.pth", type=str)
parser.add_argument('--nus_npy_path',type=str, default='/shareData3/lab-xing.jing/dataset/NUS/npy/',help='the preprocessed NUS dataset with format .npy')

# parser.add_argument('--pth_path0', default="/shareData3/lab-xing.jing/project/C4/train/log/baseline/fold0.pth", type=str)
# parser.add_argument('--pth_path1', default="/shareData3/lab-xing.jing/project/C4/train/log/baseline/fold1.pth", type=str)
# parser.add_argument('--pth_path2', default="/shareData3/lab-xing.jing/project/C4/train/log/baseline/fold2.pth", type=str)
opt = parser.parse_args()

val_loss = AverageMeter()
errors = []

#create network
SqueezeNet = squeezenet1_1(pretrained=True)
network = CreateNet(SqueezeNet).cuda()
network.eval()

max5_label_loss=0
max10_label_loss=0
max20_label_loss=0
max30_label_loss=0
max40_label_loss=0
max50_label_loss=0
all_label_loss=0

with open('/shareData3/lab-xing.jing/project/C4/train/uda_method/' + 'search.csv', 'a', newline='') as csvfile:
    header = ['loss','sum','max5','max10','max20','max30','max40','max50','var','var_weighted','light_var','S_R','S_SUM']
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(header)
    for i in range(1):
        ############################################test fold 0############################################
        dataset_test = ColorChecker(train=False,fold_num=i)
        # dataset_test = dataset_NUS(train=False, npy_path=opt.nus_npy_path)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,shuffle=False, num_workers=opt.workers)
        len_dataset_test = len(dataset_test)
        print('Len_fold:',len(dataset_test))
        if i == 0:
            pth_path = opt.pth_path0
        elif i == 1:
            pth_path = opt.pth_path1
        elif i == 2:
            pth_path = opt.pth_path2
        #load parameters
        network.load_state_dict(torch.load(pth_path))
        network.eval()
        with torch.no_grad():
            for i,data in enumerate(dataloader_test):
                img, label,fn = data
                img = Variable(img.cuda())
                img=img.pow(1 / 2.2)
                label = Variable(label.cuda())
                pred = network(img)
                pred_img = torch.nn.functional.normalize(pred,dim=1)
                # save_image_tensor(pred_img, './visualize/pred_map/'+fn[0][0:-4]+'_map.png')
                # if os.path.exists(./visualize/pred_map)
                # save_image_tensor(img, './visualize/input/'+fn[0])
                pred_ill = torch.nn.functional.normalize(torch.sum(torch.sum(pred,2),2),dim=1)
                loss = get_angular_loss(pred_ill,label)

                u_r, s_r, v_r = torch.svd(pred[:,0,:,:])
                u_sum, s_sum,v_sum = torch.svd(torch.sum(pred,dim=1))
                light_var = torch.std(torch.sum(pred,dim=1))
                sum_pred = torch.sum(pred)
                pred_ill_temp = pred.view(1,3,-1)
                pred_ill_temp_sorted = torch.sort(pred_ill_temp, dim=2, descending=True)[0]
                pred_ill_temp_5 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.05)]
                pred_ill_temp_10 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.1)]
                pred_ill_temp_20 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.2)]
                pred_ill_temp_30 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.3)]
                pred_ill_temp_40 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.4)]
                pred_ill_temp_50 = pred_ill_temp_sorted[:,:,:int(pred_ill_temp_sorted.size()[2]*0.5)]

                #sum 5%
                max5 = torch.sum(pred_ill_temp_5)
                max10 = torch.sum(pred_ill_temp_10)
                max20 = torch.sum(pred_ill_temp_20)
                max30 = torch.sum(pred_ill_temp_30)
                max40 = torch.sum(pred_ill_temp_40)
                max50 = torch.sum(pred_ill_temp_50)

                #each fake label
                fake_label_all = torch.sum(torch.sum(pred,2),2)
                fake_label_5 = torch.sum(pred_ill_temp_5,dim=2)
                fake_label_10 = torch.sum(pred_ill_temp_10,dim=2)
                fake_label_20 = torch.sum(pred_ill_temp_20,dim=2)
                fake_label_30 = torch.sum(pred_ill_temp_30,dim=2)
                fake_label_40 = torch.sum(pred_ill_temp_40,dim=2)
                fake_label_50 = torch.sum(pred_ill_temp_50,dim=2)

                #each fake label loss
                fake_label_loss_all = get_angular_loss(label,fake_label_all)
                fake_label_loss_5 = get_angular_loss(label,fake_label_5)
                fake_label_loss_10 = get_angular_loss(label,fake_label_10)
                fake_label_loss_20 = get_angular_loss(label,fake_label_20)
                fake_label_loss_30 = get_angular_loss(label,fake_label_30)
                fake_label_loss_40 = get_angular_loss(label,fake_label_40)
                fake_label_loss_50 = get_angular_loss(label,fake_label_50)

                # sum fake label loss
                max5_label_loss += fake_label_loss_5
                max10_label_loss += fake_label_loss_10
                max20_label_loss += fake_label_loss_20
                max30_label_loss += fake_label_loss_30
                max40_label_loss += fake_label_loss_40
                max50_label_loss += fake_label_loss_50
                all_label_loss += fake_label_loss_all

                var=0
                var_weighted = 0
                for n in range(pred_ill_temp.size()[2]):
                    loss_temp=get_angular_loss(pred_ill_temp[:,:,n],fake_label_all)
                    var += loss_temp
                    var_weighted+= loss_temp * torch.sum(pred_ill_temp[:, :, n])
                var = var/pred_ill_temp.size()[2]
                var_weighted = var_weighted / pred_ill_temp.size()[2]
                writer.writerow([loss.cpu().numpy(),sum_pred.cpu().numpy(),
                                 max5.cpu().numpy(),max10.cpu().numpy(),max20.cpu().numpy(),
                                 max30.cpu().numpy(),max40.cpu().numpy(),max50.cpu().numpy(),
                                 var.cpu().numpy(),var_weighted.cpu().numpy(),light_var.cpu().numpy(),
                                 s_r.cpu().numpy(),s_sum.cpu().numpy()])

                val_loss.update(loss.item())
                errors.append(loss.item())
                print('Model: %s, AE: %f'%(fn[0],loss.item()))

    mean,median,trimean,bst25,wst25,pct95 = evaluate(errors)
    print('Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f'%(mean,median,trimean,bst25,wst25,pct95))

    print('all',all_label_loss.cpu().numpy(),'5-50',max5_label_loss.cpu().numpy(),max10_label_loss.cpu().numpy(),
          max20_label_loss.cpu().numpy(),max30_label_loss.cpu().numpy(),max40_label_loss.cpu().numpy(),
          max50_label_loss.cpu().numpy(),)
