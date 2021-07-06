import os
import cv2
import math
import torch
import random
import matplotlib
import torch.nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm import trange
from option import args
from model import CFNet
from torch.optim import Adam, lr_scheduler
from dataset import MEFdataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class Train(object):
    def __init__(self):
        # configurations
        self.epoch = 1000
        self.lr = 0.000001

        # create loader
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
        self.train_set = MEFdataset(transform=self.transform)
        self.train_loader = data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # create model
        self.model = CFNet().cuda()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)

        self.Loss_list = []
        if args.validation:
            self.val_list = []
            self.best_psnr = 0

    def train(self):
        if os.path.exists(args.model_path + args.model):
            print('===>Loading pre-trained model...')
            state = torch.load(args.model_path + args.model)
            self.model.load_state_dict(state['model'])
            self.Loss_list = state['loss']
        else:
            self.Loss_list = []

        bar = tqdm(range(self.epoch))
        for ep in bar:
            loss_list = []
            i = 0
            for l_over, l_under, h_over, h_under, h in self.train_loader:
                i = i + 1
                h = (h + 1) * 127.5
                h = h.cuda()
                h_over = (h_over + 1) * 127.5
                h_over = h_over.cuda()
                h_under = (h_under + 1) * 127.5
                h_under = h_under.cuda()

                sr_over, sr_under = self.model(l_over.cuda(), l_under.cuda())

                loss = - ssim(
                    sr_over[0], h_over, win_size=7, nonnegative_ssim=True) - ssim(sr_under[0], h_under, win_size=7,
                                                                                  nonnegative_ssim=True) + 2.0
                num_CFBs = 3
                for j in range(num_CFBs):
                    loss += - ssim(sr_over[j + 1], h, win_size=7, nonnegative_ssim=True) - ssim(sr_under[j + 1], h,
                                                                                                win_size=7,
                                                                                                nonnegative_ssim=True) + 2.0

                loss_list.append(loss.item())
                bar.set_description("Epoch: %d    Loss: %.6f" % (ep, loss_list[-1]))

                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.Loss_list.append(np.mean(loss_list))

            state = {
                'model': self.model.state_dict(),
                'loss': self.Loss_list
            }

            torch.save(state, os.path.join(args.model_path, 'latest.pth'))

            if ep % 5 == 0:
                model_name = str(ep) + '.pth'
                torch.save(state, os.path.join(args.model_path, model_name))
            matplotlib.use('Agg')
            fig_train = plt.figure()
            plt.plot(self.Loss_list)
            plt.savefig('train_loss_curve.png')
            if args.validation:
                Val = Validation()
                psnr_value = Val.validation()
                self.val_list.append(psnr_value)
                if psnr_value > self.best_psnr:
                    torch.save(state, os.path.join(args.model_path, 'best_ep.pth'))
                    self.best_psnr = psnr_value
                fig_val = plt.figure()
                plt.plot(self.val_list)
                plt.savefig('val_psnr_curve.png')
            plt.close()
        print("===> Finished Training!")


class Validation(object):
    def __init__(self):
        self.psnr_list = []
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
        self.val_dir_pre = args.dir_val
        self.gt_imgs = os.listdir(self.val_dir_pre + 'gt/')
        self.over_imgs = os.listdir(self.val_dir_pre + 'lr_over/')
        self.under_imgs = os.listdir(self.val_dir_pre + 'lr_under/')
        assert len(self.over_imgs) == len(self.under_imgs)
        self.num_imgs = len(self.over_imgs)

        self.model = CFNet().cuda()
        self.state = torch.load(args.model_path + 'latest.pth')
        self.model.load_state_dict(self.state['model'])

    def validation(self):
        ep_psnr_list = []
        self.model.eval()
        with torch.no_grad():
            for idx in trange(self.num_imgs):
                img1 = cv2.imread(self.val_dir_pre + 'lr_over/' + self.over_imgs[idx])
                img1 = torch.unsqueeze(self.transform(img1), 0)
                img2 = cv2.imread(self.val_dir_pre + 'lr_under/' + self.under_imgs[idx])
                img2 = torch.unsqueeze(self.transform(img2), 0)
                img_gt = cv2.imread(self.val_dir_pre + 'gt/' + self.gt_imgs[idx])

                assert img1.shape == img2.shape

                img1 = img1.cuda()
                img2 = img2.cuda()

                sr_over, sr_under = self.model(img1, img2)
                img_fused = 0.5 * sr_over[-1] + 0.5 * sr_under[-1]
                img_fused = img_fused.squeeze(0)

                img_fused = img_fused.cpu().numpy()
                img_fused = np.transpose(img_fused, (1, 2, 0))
                img_fused = img_fused.astype(np.uint8)

                psnr_idx = self.calc_psnr(img_fused, img_gt)
                ep_psnr_list.append(psnr_idx)
        return np.mean(ep_psnr_list)

    def calc_psnr(self, img1, img2):
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        pixel_max = 1.
        return 20 * math.log10(pixel_max / math.sqrt(mse))
