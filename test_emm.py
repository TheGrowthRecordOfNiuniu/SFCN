import os
from config import Config

opt = Config('training.yml')

#gpus = ','.join([str(i) for i in opt.GPU])
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from utils.image_utils import *
from data_RGB import get_test_data, get_validation_data
# from Net import Net
from Net import Net
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchstat import stat

#model_path = './pretrained_models/model_epoch_16_loss_13.974026765674353.pth'
model_path = './pretrained_models/model_best.pth'
#model_path = './checkpoints/Deblurring/models/Net/model_epoch_28_loss_83.8104199860245.pth'
#model_path = './model_epoch_580_loss_29.69251019693911.pth'  # ./pretrained_models/OURS/model_best.pth
val_dir = './datasets/Hide/'   # 'F:/yuzhijun/Net-main/Deblurring/datasets/HIDE/'
restored_dir = './result/SFCNHide/'
model_name = 'MPR'


def save_images(images, name):
    torchvision.utils.save_image(images, name)


def test():
    # val_dir = './datasets/REDS4/'

    model_restoration = Net()
    from thop import profile
    from thop import clever_format

    x = torch.randn(1,3,720,1280)
    # y = net(x)
    macs, params = profile(model_restoration, inputs=(x, ))
    print('macs:', macs,'params:', params)
    print('--------')
    macs, params = clever_format([macs, params], "%.3f")
    print('macs:', macs,'params:', params)
    model_restoration.cuda()
    model_restoration.eval()
    print('Creat model success!')
    utils.load_checkpoint_multigpu(model_restoration, model_path)
    #model_restoration.load_state_dict(torch.load(model_path)['state_dict'] )
    print('Loading pretrained model {} success!'.format(model_path))

    val_dataset = get_validation_data(val_dir, {'patch_size': None})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)
    print('Loading test datasets success!')

    with torch.no_grad():
        use_time = []
        ssim = []
        psnr_val_rgb = []
        i = 0
        p = 0
        s = 0
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0]
            input_ = data_val[1]
            target = target.cuda()
            input_ = input_.cuda()

            with torch.no_grad():
                start_time = time.time()
                restored = model_restoration(input_)
                use_time.append(time.time() - start_time)
                img_name = data_val[2][0]

                stage_1 = restored[2]
                stage_2 = restored[1]
                restored = restored[0]
                #are =are+1
                #print("area time {}".format(are))
                print(img_name)
                i = i + 1

                for res, tar in zip(restored, target):

                    print("this pic is {}".format(i))
                    #print("this block runtima {}".format(i))
                    restored_calpsnr = utils.torchPSNR(res, tar)
                    restored_calssim = cal_ssim(res.cpu(), tar.cpu())
                    p = p+restored_calpsnr
                    s = s+restored_calssim
                    print("p :{:.4f} s{:.4f}".format(p,s))
                    psnr_val_rgb.append(restored_calpsnr)
                    #tmp_ssim = cal_ssim(res.cpu(), tar.cpu())
                    ssim.append(restored_calssim)
                    print("**********")
                    print(
                    'restored psnr: {:.4f} \t ssim: {:.4f} \t avg_psnr: {:.4f} \t avg_ssim: {:.4f} \t avg time: {:.4f}'.format(
                        psnr_val_rgb[-1], restored_calssim, torch.stack(psnr_val_rgb).mean().item(), np.array(ssim).mean(),np.array(use_time).mean()))

            dir = restored_dir + img_name + '_' + '{:.4f}'.format(psnr_val_rgb[-1])
            if not os.path.exists(dir):
               os.mkdir(dir)
            save_images(stage_1,dir + '/stage_1.jpg' )
            save_images(stage_2,dir + '/stage_2.jpg' )
            save_images(restored,dir + '/restored.jpg' )
            save_images(target,dir + '/target.jpg' )
            save_images(input_,dir + '/input.jpg' )
        avg_psnr = torch.stack(psnr_val_rgb).mean().item()
        avg_ssim = np.array(ssim).mean()
        avg_time = np.array(use_time).mean()
        print('avg_psnr: {:.4f} \t avg_ssim: {:.4f} \t avg_time: {:.4f}'.format(avg_psnr, avg_ssim, avg_time))


def tmp_test():
    model_restoration = Net()

    model_restoration.cuda()
    model_restoration.eval()
    print('Creat model success!')

    model_restoration.load_state_dict(torch.load(model_path)['state_dict'], False)
    print('Loading pretrained model {} success!'.format(model_path))

    input_files = sorted(os.listdir(val_dir + 'input'))
    target_files = sorted(os.listdir(val_dir + 'target'))

    input_files = [os.path.join(val_dir + 'input', file) for file in input_files]
    target_files = [os.path.join(val_dir + 'target', file) for file in target_files]

    with torch.no_grad():
        use_time = []
        ssim = []
        psnr_val_rgb = []
        for input_, target in zip(input_files, target_files):
            name = input_
            input_ = Image.open(input_)
            target = Image.open(target)
            input_ = TF.to_tensor(input_).cuda()
            target = TF.to_tensor(target).cuda()

            with torch.no_grad():
                start_time = time.time()
                model_restoration(input_)
                use_time.append(time.time() - start_time)
                img_name = name.split("\\")[-1]

                stage_1 = restored[2]
                stage_2 = restored[1]
                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    tmp_ssim = cal_ssim(res.cpu(), tar.cpu())
                    ssim.append(tmp_ssim)
                    print('psnr: {:.4f} \t ssim: {:.4f} '.format(psnr_val_rgb[-1], tmp_ssim))

            dir = restored_dir + img_name + '_' + '{:.4f}'.format(psnr_val_rgb[-1])
            if not os.path.exists(dir):
                os.mkdir(dir)
            save_images(stage_1, dir + '/stage_1.jpg')
            save_images(stage_2, dir + '/stage_2.jpg')
            save_images(restored, dir + '/restored.jpg')
            save_images(target, dir + '/target.jpg')
            save_images(input_, dir + '/input.jpg')
        avg_psnr = torch.stack(psnr_val_rgb).mean().item()
        avg_ssim = np.array(ssim).mean()
        avg_time = np.array(use_time).mean()
        print('{} \t avg_psnr: {:.4f} \t avg_ssim: {:.4f} \t avg_time: {:.4f}'.format(model_name, avg_psnr, avg_ssim,
                                                                                      avg_time))


if __name__ == '__main__':
    # tmp_test()
    test()
