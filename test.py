import os
import cv2
import time
import torch
import torch.nn
import numpy as np
import torchvision.transforms as transforms

from tqdm import trange
from model import CFNet
from option import args


class Test:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
        self.test_dir_pre = args.dir_test
        self.over_imgs = os.listdir(self.test_dir_pre + 'lr_over/')
        self.over_imgs.sort()
        self.under_imgs = os.listdir(self.test_dir_pre + 'lr_under/')
        self.under_imgs.sort()
        assert len(self.over_imgs) == len(self.under_imgs)
        self.num_imgs = len(self.over_imgs)

        self.model = CFNet().cuda()
        self.state = torch.load(args.model_path + args.model)
        self.model.load_state_dict(self.state['model'])

        self.test_time = []

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for idx in trange(1, 1 + self.num_imgs):
                img1 = cv2.imread(self.over_imgs[idx])
                img1 = torch.unsqueeze(self.transform(img1), 0)
                img2 = cv2.imread(self.under_imgs[idx])
                img2 = torch.unsqueeze(self.transform(img2), 0)

                assert img1.shape == img2.shape
                save_name = os.path.splitext(os.path.split(self.over_imgs[idx])[1])[0]

                img1 = img1.cuda()
                img2 = img2.cuda()
                torch.cuda.synchronize()
                start_time = time.time()

                sr_over, sr_under = self.model(img1, img2)
                img_fused = 0.5 * sr_over[-1] + 0.5 * sr_under[-1]
                img_fused = img_fused.squeeze(0)

                torch.cuda.synchronize()
                end_time = time.time()
                self.test_time.append(end_time - start_time)

                img_fused = img_fused.cpu().numpy()
                img_fused = np.transpose(img_fused, (1, 2, 0))
                img_fused = img_fused.astype(np.uint8)

                cv2.imwrite(os.path.join(args.save_dir, str(save_name) + args.ext), img_fused)

            print('The average testing time is {:.4f} s.'.format(np.mean(self.test_time)))
