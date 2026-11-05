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
    """PyTorch Dataset that supports one LR channel and two HR channels (VV & VH) plus ancillary layers (IncAngle, EWC, DEM).

    Returns 6 elements per sample:
      (LR, HR_VV, HR_VH, Angle, LC, DEM)
    """

    def __init__(
        self,
        data_root="sr_data",
        mode="train",
        indices=None,
        zscore=True,
        augment=True,
        means=None,
        stds=None,
    ):
        self.mode = mode
        self.zscore = zscore
        self.augment = augment
        self.lr_root = os.path.join(data_root, "LR")
        self.hr_roots = [
            os.path.join(data_root, "HR", "VV"),
            os.path.join(data_root, "HR", "VH"),
        ]
        self.angle_root = os.path.join(data_root, "IncAngle")
        self.ewc_root = os.path.join(data_root, "EWC")
        self.dem_root = os.path.join(data_root, "DEM")

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

        self._n_layers = 6
        self.samples = []
        self._build_samples()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # Reuse the training set's statistics (e.g. for validation) when given
        if self.zscore and means is not None and stds is not None:
            self._means = means
            self._stds = stds
        elif self.zscore:
            self._compute_zscore_stats()

    @staticmethod
    def _extract_timestamp(filename):
        match = re.search(r"(20\d{6})", filename)
        return match.group(1) if match else None

    def _find_matching_file(self, folder, timestamp):
        for fname in os.listdir(folder):
            if timestamp in fname:
                return fname
        return None

    def _build_samples(self):
        for region_folder in sorted(os.listdir(self.lr_root)):
            lr_dir = os.path.join(self.lr_root, region_folder)
            if not os.path.isdir(lr_dir):
                continue

            region = region_folder.replace("-SMAP", "")
            hr_dirs = [os.path.join(hr_root, f"{region}-{'VV' if i==0 else 'VH'}") for i, hr_root in enumerate(self.hr_roots)]
            angle_dir = os.path.join(self.angle_root, f"{region}-Ang")
            ewc_path = os.path.join(self.ewc_root, f"EWC556{region}.tif")
            dem_path = os.path.join(self.dem_root, f"DEM556{region}.tif")

            for lr_file in sorted(os.listdir(lr_dir)):
                timestamp = self._extract_timestamp(lr_file)
                if timestamp is None:
                    continue

                hr_files = []
                for hr_dir in hr_dirs:
                    hr_match = self._find_matching_file(hr_dir, timestamp)
                    if hr_match is None:
                        break
                    hr_files.append(os.path.join(hr_dir, hr_match))
                else:
                    angle_match = self._find_matching_file(angle_dir, timestamp)
                    if angle_match:
                        self.samples.append(
                            (
                                region,
                                os.path.join(lr_dir, lr_file),
                                hr_files,
                                os.path.join(angle_dir, angle_match),
                                ewc_path,
                                dem_path,
                            )
                        )


    def _compute_zscore_stats(self):
        means = [[] for _ in range(self._n_layers)]
        stds = [[] for _ in range(self._n_layers)]

        for (_, lr_p, hr_ps, angle_p, lc_p, dem_p) in self.samples:
            paths = [lr_p] + hr_ps + [angle_p, lc_p, dem_p]
            for i, p in enumerate(paths):
                arr = rasterio.open(p).read().squeeze().astype(np.float32)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    means[i].append(arr.mean())
                    stds[i].append(arr.std())

        self._means = [float(np.mean(m)) if m else 0.0 for m in means]
        self._stds = [float(np.mean(s)) if s else 1.0 for s in stds]

    def _zscore_normalize(self, img, idx):
        img = np.nan_to_num(img)
        mean = self._means[idx]
        std = self._stds[idx]
        return (img - mean) / std if std else img - mean

    @staticmethod
    def _random_flip_and_rotate(imgs):
        if random.random() > 0.5:
            imgs = [np.fliplr(im) for im in imgs]
        if random.random() > 0.5:
            imgs = [np.flipud(im) for im in imgs]
        k = random.randint(0, 3)
        if k:
            imgs = [np.rot90(im, k) for im in imgs]
        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        region, lr_path, hr_paths, angle_path, lc_path, dem_path = self.samples[idx]

        im_lr = rasterio.open(lr_path).read().squeeze().astype(np.float32)
        im_hr_list = [rasterio.open(p).read().squeeze().astype(np.float32) for p in hr_paths]
        im_angle = rasterio.open(angle_path).read().squeeze().astype(np.float32)
        im_lc = rasterio.open(lc_path).read().squeeze().astype(np.float32)
        im_dem = rasterio.open(dem_path).read().squeeze().astype(np.float32)

        def _crop_resize(im):
            return cv2.resize(im[3:256, 2:253], (256, 256), interpolation=cv2.INTER_NEAREST)

        im_lr = _crop_resize(im_lr)
        im_hr_list = [_crop_resize(im) for im in im_hr_list]
        im_angle = _crop_resize(im_angle)
        im_lc = _crop_resize(im_lc)
        im_dem = _crop_resize(im_dem)

        im_lr = self._remove_region_parts(im_lr, region)
        im_hr_list = [self._remove_region_parts(im, region) for im in im_hr_list]
        im_angle = self._remove_region_parts(im_angle, region)
        im_lc = self._remove_region_parts(im_lc, region)
        im_dem = self._remove_region_parts(im_dem, region)

        if self.mode == "train" and self.augment:
            imgs_aug = self._random_flip_and_rotate([im_lr] + im_hr_list + [im_angle, im_lc, im_dem])
            im_lr = imgs_aug[0]
            im_hr_list = imgs_aug[1:1 + len(im_hr_list)]
            im_angle, im_lc, im_dem = imgs_aug[-3:]

        im_lr = cv2.resize(im_lr, (256, 256), interpolation=cv2.INTER_NEAREST)
        im_hr_list = [cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST) for im in im_hr_list]
        im_angle = cv2.resize(im_angle, (256, 256), interpolation=cv2.INTER_NEAREST)
        im_lc = cv2.resize(im_lc, (256, 256), interpolation=cv2.INTER_NEAREST)
        im_dem = cv2.resize(im_dem, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.zscore:
            im_lr = self._zscore_normalize(im_lr, 0)
            for i in range(len(im_hr_list)):
                im_hr_list[i] = self._zscore_normalize(im_hr_list[i], 1 + i)
            im_angle = self._zscore_normalize(im_angle, 1 + len(im_hr_list))
            im_lc = self._zscore_normalize(im_lc, 2 + len(im_hr_list))
            im_dem = self._zscore_normalize(im_dem, 3 + len(im_hr_list))

        return (
            im_lr[None],
            im_hr_list[0][None],  # HR VV
            im_hr_list[1][None],  # HR VH
            im_angle[None],
            im_lc[None],
            im_dem[None],
        )

    def _remove_region_parts(self, img, region):
        rule = self.remove_rules.get(region)
        if rule is None:
            return img
        if "rows" in rule:
            img = np.delete(img, np.arange(rule["rows"].start, rule["rows"].stop), axis=0)
        if "cols" in rule:
            img = np.delete(img, rule["cols"], axis=1)
        return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = LRHRDataSet(data_root="sr_data", mode="train", zscore=False, augment=False)
    os.makedirs("sanity_plots_new_2HR", exist_ok=True)

    seen_regions = set()
    for i in range(len(dataset)):
        region = dataset.samples[i][0]
        if region in seen_regions:
            continue

        seen_regions.add(region)
        im_lr, im_hr_vv, im_hr_vh, im_angle, im_lc, im_dem = dataset[i]

        images = {
            "LR": im_lr.squeeze(),
            "HR VV": im_hr_vv.squeeze(),
            "HR VH": im_hr_vh.squeeze(),
            "Angle": im_angle.squeeze(),
            "Land Cover": im_lc.squeeze(),
            "DEM": im_dem.squeeze(),
        }

        print(f"\nSample {i} | Region: {region}")
        for name, img in images.items():
            nan_mask = np.isnan(img)
            if nan_mask.any() and name.startswith("HR"):
                nan_rows = np.unique(np.where(nan_mask)[0])
                nan_cols = np.unique(np.where(nan_mask)[1])
                print(f"  {name:<12} | NaN Rows: {nan_rows.tolist()} | NaN Cols: {nan_cols.tolist()}")

            images[name] = np.nan_to_num(img)

        # Plotting
        plt.figure(figsize=(18, 3))
        for j, (name, img) in enumerate(images.items()):
            plt.subplot(1, 6, j + 1)
            if name in ["LR", "HR VV", "HR VH"]:
                cmap = "gray"
            elif name == "Angle":
                cmap = "viridis"
            elif name == "Land Cover":
                cmap = "tab20"
            else:
                cmap = "terrain"
            plt.imshow(img, cmap=cmap)
            plt.title(name)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"sanity_plots_new_2HR/sample_{i}_{region}.png")
        plt.close()
