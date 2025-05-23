import cv2
import numpy as np
from torch.utils import data
import kornia
import rasterio
import os
import numpy as np
import torch.nn as nn
import torch


class LRHRDataSet(data.Dataset):
    def __init__(
        self,
        data_root="SR-Data",  # source folder e.g. SR-Trial/Nu21L556,
        mode="train",
        indices=None,
    ):
        # -----------------loading data
        self.mode = mode
        subfolders = ["Nu621P1"]  # Adjust to the current folder structure
        low_files, high_files, cond1_files, cond2_files, cond3_files = (
            [],
            [],
            [],
            [],
            [],
        )  # conditional data amount depends on experiment
        for subfold in subfolders:
            low_path = os.path.join(data_root, "LR", f"{subfold}-SMAP")
            files = sorted(os.listdir(low_path))
            low_files_temp = [os.path.join(low_path, x) for x in files if ".tif" in x]
            low_files = low_files + low_files_temp

            high_path = os.path.join(data_root, "HR", "VV", f"{subfold}-VV")
            files = sorted(os.listdir(high_path))
            high_files_temp = [os.path.join(high_path, x) for x in files if ".tif" in x]
            high_files = high_files + high_files_temp

            cond1_path = os.path.join(data_root, "Condi", "DEM")#, f"DEM556{subfold}.tif")
            if os.path.isdir(cond1_path):
                cond1_files = os.listdir(cond1_path)
            else:
                cond1_files = [cond1_path]
            cond1_files_temp = [
                os.path.join(cond1_path, cond1_files[0])
                for i in range(len(high_files_temp))
            ]
            cond1_files = cond1_files + cond1_files_temp

            cond2_path = os.path.join(data_root, "Condi", "EWC")#, f"EWC556{subfold}.tif")
            if os.path.isdir(cond2_path):
                cond2_files = os.listdir(cond2_path)
            else:
                cond2_files = [cond2_path]
            cond2_files_temp = [
                os.path.join(cond2_path, cond2_files[0])
                for i in range(len(high_files_temp))
            ]
            cond2_files = cond2_files + cond2_files_temp

        if indices is None:
            self.low_files = np.array(low_files)
            self.high_files = np.array(high_files)
            self.cond1_files = np.array(cond1_files)
            self.cond2_files = np.array(cond2_files)
        else:
            self.low_files = np.array(low_files)[indices]
            self.high_files = np.array(high_files)[indices]
            self.cond1_files = np.array(cond1_files)[indices]
            self.cond2_files = np.array(cond2_files)[indices]

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        im_lr = rasterio.open(self.low_files[idx]).read().squeeze()  # 200-330
        im_hr = rasterio.open(self.high_files[idx]).read().squeeze()  # -15- -4
        im_cond1 = (
            rasterio.open(self.cond1_files[idx]).read().squeeze()
        )  
        im_cond2 = rasterio.open(self.cond2_files[idx]).read().squeeze()

        # crop nans:
        im_lr, im_hr, im_cond1, im_cond2 = (
            im_lr[3:256, 2:253],
            im_hr[3:256, 2:253],
            im_cond1[3:256, 2:253],
            im_cond2[3:256, 2:253],
        )
        tgt_size = (256, 256)
        im_lr = cv2.resize(im_lr, tgt_size, interpolation=cv2.INTER_NEAREST)
        im_hr = cv2.resize(im_hr, tgt_size, interpolation=cv2.INTER_NEAREST)
        im_cond1 = cv2.resize(im_cond1, tgt_size, interpolation=cv2.INTER_NEAREST)
        im_cond2 = cv2.resize(im_cond2, tgt_size, interpolation=cv2.INTER_NEAREST)

        return im_lr[None], im_hr[None], im_cond1[None], im_cond2[None]


class DataAugmenter:
    def __init__(self):
        self.transform = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomVerticalFlip(),
        )

    def augment_sample(self, im_lr, im_hr, im_cond1, im_cond2):
        data_concat = torch.cat([im_lr, im_hr, im_cond1, im_cond2], dim=1)
        data_concat = self.transform(data_concat)
        im_lr, im_hr, im_cond1, im_cond2 = (
            data_concat[:, 0],
            data_concat[:, 1],
            data_concat[:, 2],
            data_concat[:, 3],
        )
        return im_lr[:, None], im_hr[:, None], im_cond1[:, None], im_cond2[:, None]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dataset = LRHRDataSet(mode="train")

    augmenter = DataAugmenter()
    trainloader = data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True
    )
    for i_iter, batch in enumerate(trainloader):

        im_lr, im_hr, im_cond1, im_cond2 = batch
        print(im_lr.shape)
        print(im_hr.shape)
        print(im_cond1.shape)

        print(im_lr.shape)
        print(im_hr.shape)
        print(im_cond1.shape)
        im_lr, im_hr, im_cond1 = (
            im_lr.detach().cpu().numpy(),
            im_hr.detach().cpu().numpy(),
            im_cond1.detach().cpu().numpy(),
        )
        for j in range(4):
            plt.figure()
            plt.subplot(131)
            plt.imshow(im_lr[j].squeeze())
            plt.subplot(132)
            plt.imshow(im_hr[j].squeeze())

            plt.subplot(133)
            plt.imshow(im_cond1[j].squeeze())
            plt.show()
