import os
import cv2
import random
import torch.utils.data as data

from option import args


class MEFdataset(data.Dataset):
    def __init__(self, transform):
        super(MEFdataset, self).__init__()
        self.dir_prefix = args.dir_train
        self.lr_over = os.listdir(self.dir_prefix + 'lr_over/')
        self.lr_over.sort()
        self.lr_under = os.listdir(self.dir_prefix + 'lr_under/')
        self.lr_under.sort()
        self.hr_over = os.listdir(self.dir_prefix + 'hr_over/')
        self.hr_over.sort()
        self.hr_under = os.listdir(self.dir_prefix + 'hr_under/')
        self.hr_under.sort()
        self.hr = os.listdir(self.dir_prefix + 'hr/')
        self.hr.sort()

        self.scale = args.scale
        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):
        lr_over = cv2.imread(self.dir_prefix + 'lr_over/' + self.lr_over[idx])
        lr_under = cv2.imread(self.dir_prefix + 'lr_under/' + self.lr_under[idx])
        hr_over = cv2.imread(self.dir_prefix + 'hr_over/' + self.hr_over[idx])
        hr_under = cv2.imread(self.dir_prefix + 'hr_under/' + self.hr_under[idx])
        hr = cv2.imread(self.dir_prefix + 'hr/' + self.hr[idx])

        lr_over_p, lr_under_p, hr_over_p, hr_under_p, hr_p = self.get_patch(lr_over,
                                                                            lr_under,
                                                                            hr_over,
                                                                            hr_under,
                                                                            hr)
        if self.transform:
            lr_over_p = self.transform(lr_over_p)
            lr_under_p = self.transform(lr_under_p)
            hr_over_p = self.transform(hr_over_p)
            hr_under_p = self.transform(hr_under_p)
            hr_p = self.transform(hr_p)

        return lr_over_p, lr_under_p, hr_over_p, hr_under_p, hr_p

    def get_patch(self, l_over, l_under, h_over, h_under, h):
        lh, lw = l_over.shape[:2]
        l_stride = self.patch_size
        scale = self.scale
        h_stride = l_stride * scale

        x = random.randint(0, lw - l_stride)
        y = random.randint(0, lh - l_stride)
        ox = scale * x
        oy = scale * y

        l_over = l_over[y:y + l_stride, x:x + l_stride, :]
        l_under = l_under[y:y + l_stride, x:x + l_stride, :]
        h_over = h_over[oy:oy + h_stride, ox:ox + h_stride, :]
        h_under = h_under[oy:oy + h_stride, ox:ox + h_stride, :]
        h = h[oy:oy + h_stride, ox:ox + h_stride, :]

        return l_over, l_under, h_over, h_under, h
