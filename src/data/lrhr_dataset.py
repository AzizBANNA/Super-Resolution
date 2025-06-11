import os
import re
import random
import rasterio
import numpy as np
import cv2
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)


class LRHRDataSet(Dataset):
    def __init__(self, data_root="sr_data", mode="train", indices=None, zscore=True, augment=True):
        self.mode = mode
        self.zscore = zscore
        self.augment = augment
        self.lr_root = os.path.join(data_root, "LR")
        self.hr_root = os.path.join(data_root, "HR", "VV")
        self.angle_root = os.path.join(data_root, "IncAngle")
        self.ewc_root = os.path.join(data_root, "EWC")
        self.dem_root = os.path.join(data_root, "DEM")
        self.printed_regions = set()

        self.remove_rules = {
            "Nu621P2": {"rows": slice(0, 17), "cols": [255]},
            "Nu621P3": {"cols": [0, 1]},
            "Nu622P1": {"cols": [0, 1]},
            "Nu622P2": {"rows": slice(0, 9), "cols": [255]},
            "Nu622P3": {"cols": [0, 1, 2, 3]},
            "Nu623P1": {"cols": list(range(0, 4))},
            "Nu623P2": {"cols": [0, 1]},
            "Nu623P3": {"cols": list(range(0, 6))},
            "Nu911P1": {"rows": slice(0, 16), "cols": list(range(0, 14)) + [255]},
            "Nu911P2": {"rows": slice(0, 20), "cols": list(range(0, 16))},
            "Nu911P3": {"rows": slice(0, 14), "cols": list(range(0, 11))},
            "Nu912P1": {"rows": slice(0, 16), "cols": list(range(0, 11)) + [253, 255]},
            "Nu912P2": {"rows": slice(0, 17), "cols": list(range(0, 13))},
            "Nu912P3": {"rows": slice(0, 13), "cols": list(range(0, 11)) + [255]},
            "Nu913P1": {"rows": slice(0, 17), "cols": list(range(0, 12)) + [255]},
            "Nu913P2": {"rows": slice(0, 16), "cols": list(range(0, 11))},
            "Nu913P3": {"rows": slice(0, 14), "cols": list(range(0, 9)) + [255]},
        }

        self.samples = []
        self.build_samples()
        self.samples = np.array(self.samples if indices is None else np.array(self.samples)[indices])

        if self.zscore:
            self._compute_zscore_stats()

    def extract_timestamp(self, filename):
        match = re.search(r"(20\d{6})", filename)
        return match.group(1) if match else None

    def build_samples(self):
        for region_folder in sorted(os.listdir(self.lr_root)):
            if not os.path.isdir(os.path.join(self.lr_root, region_folder)):
                continue
            region = region_folder.replace("-SMAP", "")
            lr_dir = os.path.join(self.lr_root, region_folder)
            hr_dir = os.path.join(self.hr_root, f"{region}-VV")
            angle_dir = os.path.join(self.angle_root, f"{region}-Ang")
            ewc_path = os.path.join(self.ewc_root, f"EWC556{region}.tif")
            dem_path = os.path.join(self.dem_root, f"DEM556{region}.tif")

            for lr_file in sorted(os.listdir(lr_dir)):
                timestamp = self.extract_timestamp(lr_file)
                if timestamp is None:
                    continue

                hr_file = self.find_matching_file(hr_dir, timestamp)
                angle_file = self.find_matching_file(angle_dir, timestamp)

                if hr_file and angle_file:
                    self.samples.append(
                        (
                            region,
                            os.path.join(lr_dir, lr_file),
                            os.path.join(hr_dir, hr_file),
                            os.path.join(angle_dir, angle_file),
                            ewc_path,
                            dem_path,
                        )
                    )

    def find_matching_file(self, folder, timestamp):
        for fname in os.listdir(folder):
            if timestamp in fname:
                return fname
        return None

    def remove_region_parts(self, img, region):
        rule = self.remove_rules.get(region)
        if rule is None:
            return img
        if "rows" in rule:
            img = np.delete(img, np.arange(rule["rows"].start, rule["rows"].stop), axis=0)
        if "cols" in rule:
            img = np.delete(img, rule["cols"], axis=1)
        return img

    def _compute_zscore_stats(self):
        means = [[] for _ in range(5)]
        stds = [[] for _ in range(5)]

        for region, lr_path, hr_path, angle_path, lc_path, dem_path in self.samples:
            paths = [lr_path, hr_path, angle_path, lc_path, dem_path]
            for i, p in enumerate(paths):
                arr = rasterio.open(p).read().squeeze().astype(np.float32)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    means[i].append(arr.mean())
                    stds[i].append(arr.std())

        self._means = [float(np.mean(m)) if m else 0.0 for m in means]
        self._stds = [float(np.mean(s)) if s else 1.0 for s in stds]

    def zscore_normalize(self, img, idx):
        img = np.nan_to_num(img)
        mean = self._means[idx]
        std = self._stds[idx]
        return (img - mean) / std if std != 0 else img - mean

    def _random_flip_and_rotate(self, ims):
        if random.random() > 0.5:
            ims = [np.fliplr(im) for im in ims]
        if random.random() > 0.5:
            ims = [np.flipud(im) for im in ims]
        k = random.randint(0, 3)
        if k > 0:
            ims = [np.rot90(im, k) for im in ims]
        return ims

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        region, lr_path, hr_path, angle_path, lc_path, dem_path = self.samples[idx]
        im_lr = rasterio.open(lr_path).read().squeeze().astype(np.float32)
        im_hr = rasterio.open(hr_path).read().squeeze().astype(np.float32)
        im_angle = rasterio.open(angle_path).read().squeeze().astype(np.float32)
        im_lc = rasterio.open(lc_path).read().squeeze().astype(np.float32)
        im_dem = rasterio.open(dem_path).read().squeeze().astype(np.float32)

        # Generic crop and resize
        im_lr, im_hr, im_angle, im_lc, im_dem = [
            cv2.resize(im[3:256, 2:253], (256, 256), interpolation=cv2.INTER_NEAREST)
            for im in [im_lr, im_hr, im_angle, im_lc, im_dem]
        ]

        # Region-specific removal
        im_lr = self.remove_region_parts(im_lr, region)
        im_hr = self.remove_region_parts(im_hr, region)
        im_angle = self.remove_region_parts(im_angle, region)
        im_lc = self.remove_region_parts(im_lc, region)
        im_dem = self.remove_region_parts(im_dem, region)

        # Optional data augmentation (train mode only)
        if self.mode == "train" and self.augment:
            im_lr, im_hr, im_angle, im_lc, im_dem = self._random_flip_and_rotate(
                [im_lr, im_hr, im_angle, im_lc, im_dem]
            )

        # Final resize after removal
        im_lr, im_hr, im_angle, im_lc, im_dem = [
            cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
            for im in [im_lr, im_hr, im_angle, im_lc, im_dem]
        ]

        # Z-score normalization
        if self.zscore:
            im_lr = self.zscore_normalize(im_lr, 0)
            im_hr = self.zscore_normalize(im_hr, 1)
            im_angle = self.zscore_normalize(im_angle, 2)
            im_lc = self.zscore_normalize(im_lc, 3)
            im_dem = self.zscore_normalize(im_dem, 4)

        return im_lr[None], im_hr[None], im_angle[None], im_lc[None], im_dem[None]
