import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
import pydicom as dicom
from time import time
import skimage.exposure as hist
from os import makedirs
from os.path import exists, join
from datetime import datetime
from torchvision import transforms
from PIL import Image
from visdom import Visdom
from pathlib import Path
import os
import sys

sys.path.insert(0, "../model/")
from sklearn.preprocessing import OneHotEncoder


class DeidCXRDataset(Dataset):

    def __init__(
        self,
        root_path,
        metadata_df_path,
        label_cols,
        cont_label_cols=[],
        multi_view=False,
        include_demo=False,
        downsample_size=224,
        adaptive_norm=True,
        equalize_norm=False,
        brighten=False,
        gaussian_noise=True,
        gaussian_noise_settings=(0, 0.05),
        center_cropped=False,
        center_cropped_size=1400,
        multi_resolution=False,
        multi_resolution_sizes=["full", 1400, 900],
        rand_affine=False,
        age_stats=None,
        enc=None,
        train=False,
        handle_nans="replace_zero",
        handle_minus_one="replace_zero",
        filter_by_first_cxr=False,
        filter_by_cxr_before_echo=False,
        optuna_trial=None,
    ):

        self.metadata = pd.read_csv(metadata_df_path)
        self.root_path = root_path
        self.label_cols = label_cols
        self.cont_label_cols = cont_label_cols

        # Unique deid for each row
        self.metadata["IMAGE_ID"] = np.unique(
            self.metadata["cxr_filename"], return_inverse=True
        )[-1]

        self.mapping = self.metadata[["IMAGE_ID", "cxr_filename"]]

        self.metadata.set_index("IMAGE_ID", inplace=True)
        self.train = train
        self.multi_view = multi_view
        self.include_demo = include_demo
        self.downsample_size = downsample_size
        self.adaptive_norm = adaptive_norm
        self.equalize_norm = equalize_norm
        self.brighten = brighten
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_settings = gaussian_noise_settings
        self.center_cropped = center_cropped
        self.center_cropped_size = center_cropped_size
        self.multi_resolution = multi_resolution
        self.multi_resolution_sizes = multi_resolution_sizes
        self.rand_affine = rand_affine
        self.age_stats = age_stats
        self.enc = enc

        if self.include_demo:
            demo_vars = self.metadata[["sex", "age"]].to_numpy()

            if enc is None:
                enc = OneHotEncoder()
                self.encoded_sex = enc.fit_transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()
                self.enc = enc
            else:
                self.enc = enc
                self.encoded_sex = self.enc.transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()

            self.locs_to_ilocs = dict(
                zip(self.metadata.index, np.arange(len(self.metadata.index)))
            )

            if age_stats is None:
                self.age_mean = self.metadata["age"].mean()
                self.age_var = self.metadata["age"].var()
                self.metadata["age"] = (
                    self.metadata["age"] - self.metadata["age"].mean()
                ) / np.sqrt(self.metadata["age"].var())
            else:
                self.metadata["age"] = (self.metadata["age"] - age_stats[0]) / np.sqrt(
                    age_stats[1]
                )

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]

        if self.include_demo:
            demo_index = self.locs_to_ilocs[item]
            encoded_sex = self.encoded_sex[demo_index]
            encoded_age = image_metadata["age"]
            demo_tensor = torch.from_numpy(
                np.concatenate((encoded_sex, np.array([encoded_age])))
            )
            demo_tensor = demo_tensor.float()

        path = image_metadata["cxr_path"]

        path = os.path.join(self.root_path, path)

        try:
            image_array = [cv2.imread(path)[:, :, 0]]
        except:
            print("Error in reading file {}".format(path))
            return None

        if self.train and self.gaussian_noise:
            mu, std = self.gaussian_noise_settings
            image_array = [
                ds
                + np.random.normal(
                    mu, std, size=(self.downsample_size, self.downsample_size)
                )
                for ds in image_array
            ]

        image_array = [torch.from_numpy(ds).float() for ds in image_array]

        if len(image_array) == 1:
            image_array = image_array[0].unsqueeze(0)
            image_array = image_array.repeat(3, 1, 1)
        else:
            image_array = torch.stack(image_array)

        labels = image_metadata[self.label_cols]
        labels = torch.tensor(labels)

        cont_labels = torch.tensor([])
        if len(self.cont_label_cols) > 0:
            cont_labels = image_metadata[self.cont_label_cols]
            cont_labels = torch.tensor(cont_labels)

        if self.include_demo:
            return image_array, demo_tensor, labels, cont_labels
        else:
            return image_array, labels, cont_labels


class CXRDatasetNumpy(Dataset):

    def __init__(
        self,
        metadata_df_path,
        cxr_data_path,
        tabular_data_path,
        index_identifier,
        label_cols,
        cont_label_cols=[],
        gaussian_noise=True,
        gaussian_noise_settings=(0, 0.05),
        train=True,
        **kwargs
    ):

        self.metadata = pd.read_csv(metadata_df_path)
        self.cxr_data = np.load(cxr_data_path)
        self.tabular_data = np.load(tabular_data_path)
        self.label_cols = label_cols
        self.cont_label_cols = cont_label_cols
        self.labels = self.metadata[label_cols].to_numpy()
        self.cont_labels = self.metadata[cont_label_cols].to_numpy()
        self.train = train
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_settings = gaussian_noise_settings

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        dat = (torch.from_numpy(self.cxr_data), torch.from_numpy(self.tabular_data))
        labs = (torch.from_numpy(self.cont_labels), torch.from_numpy(self.labels))

        return dat, labs


class CXRDataset(Dataset):

    def __init__(
        self,
        metadata_df_path,
        pat_identifier,
        label_cols,
        cont_label_cols=[],
        multi_view=False,
        include_demo=False,
        downsample_size=224,
        adaptive_norm=True,
        equalize_norm=False,
        brighten=False,
        gaussian_noise=True,
        gaussian_noise_settings=(0, 0.05),
        center_cropped=False,
        center_cropped_size=1400,
        multi_resolution=False,
        multi_resolution_sizes=["full", 1400, 900],
        rand_affine=False,
        age_stats=None,
        enc=None,
        train=False,
        handle_nans="replace_zero",
        handle_minus_one="replace_zero",
        filter_by_first_cxr=False,
        filter_by_cxr_before_echo=False,
        optuna_trial=None,
    ):

        self.metadata = pd.read_csv(metadata_df_path)
        self.handle_nans = handle_nans
        self.handle_minus_one = handle_minus_one

        if self.handle_nans == "replace_zero":
            self.metadata[label_cols] = self.metadata[label_cols].fillna(0.0)
        elif self.handle_nans == "replace_ones":
            self.metadata[label_cols] = self.metadata[label_cols].fillna(1.0)
        elif self.handle_nans == "remove":
            self.metadata = self.metadata.dropna(subset=label_cols)

        if self.handle_minus_one == "replace_zero":
            self.metadata[label_cols] = self.metadata[label_cols].replace(-1.0, 0.0)
        elif self.handle_minus_one == "replace_ones":
            self.metadata[label_cols] = self.metadata[label_cols].replace(-1.0, 1.0)
        elif self.handle_minus_one == "remove":
            self.metadata = self.metadata[
                (self.metadata[label_cols] == -1.0).any(axis=1)
            ]

        self.label_cols = label_cols
        self.cont_label_cols = cont_label_cols

        self.metadata["IMAGE_ID"] = np.unique(
            self.metadata["filename"], return_inverse=True
        )[-1]

        self.mapping = self.metadata[["IMAGE_ID", "filename"]]

        self.metadata.set_index("IMAGE_ID", inplace=True)
        self.train = train
        self.multi_view = multi_view
        self.include_demo = include_demo
        self.downsample_size = downsample_size
        self.adaptive_norm = adaptive_norm
        self.equalize_norm = equalize_norm
        self.brighten = brighten
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_settings = gaussian_noise_settings
        self.center_cropped = center_cropped
        self.center_cropped_size = center_cropped_size
        self.multi_resolution = multi_resolution
        self.multi_resolution_sizes = multi_resolution_sizes
        self.rand_affine = rand_affine
        self.age_stats = age_stats
        self.enc = enc

        if self.include_demo:
            demo_vars = self.metadata[["Sex", "age"]].to_numpy()

            if enc is None:
                enc = OneHotEncoder()
                self.encoded_sex = enc.fit_transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()
                self.enc = enc
            else:
                self.enc = enc
                self.encoded_sex = self.enc.transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()

            self.locs_to_ilocs = dict(
                zip(self.metadata.index, np.arange(len(self.metadata.index)))
            )

            if age_stats is None:
                self.age_mean = self.metadata["age"].mean()
                self.age_var = self.metadata["age"].var()
                self.metadata["age"] = (
                    self.metadata["age"] - self.metadata["age"].mean()
                ) / np.sqrt(self.metadata["age"].var())
            else:
                self.metadata["age"] = (self.metadata["age"] - age_stats[0]) / np.sqrt(
                    age_stats[1]
                )

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]

        if self.include_demo:
            demo_index = self.locs_to_ilocs[item]
            encoded_sex = self.encoded_sex[demo_index]
            encoded_age = image_metadata["age"]
            demo_tensor = torch.from_numpy(
                np.concatenate((encoded_sex, np.array([encoded_age])))
            )
            demo_tensor = demo_tensor.float()

        path = image_metadata["path"]

        try:
            image_array = dicom.read_file(path).pixel_array
            image_array = [image_array]
        except:
            print("Error in reading DICOM file {}".format(path))
            return None

        if self.multi_resolution:
            image_array = Image.fromarray(image_array[0])
            multi_res = self.multi_resolution_sizes

            cropped_images = []
            for res in multi_res:
                if res == "full":
                    cropped_images.append(np.array(image_array))
                else:
                    center_crop = transforms.Compose([transforms.CenterCrop(res)])
                    image_array = center_crop(image_array)
                    cropped_images.append(np.array(image_array))

            image_array = cropped_images

        if self.center_cropped:
            image_array = Image.fromarray(image_array[0])
            center_crop = transforms.Compose(
                [transforms.CenterCrop(self.center_cropped_size)]
            )
            image_array = center_crop(image_array)
            image_array = [np.array(image_array)]

        downsampled = [
            zoom_2D(im_arr, (self.downsample_size, self.downsample_size))
            for im_arr in image_array
        ]

        if self.adaptive_norm:
            downsampled = [adaptive_normalization_param(ds) for ds in downsampled]

        # if self.equalize_norm:
        #     downsampled = [equalize_normalization(ds) for ds in downsampled]

        if self.brighten:
            downsampled = [brighten(ds) for ds in downsampled]

        if self.train and self.rand_affine:
            rand_aff = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.RandomAffine(
                                degrees=0,
                                translate=(0.05, 0.05),
                                scale=(0.95, 1.05),
                                fillcolor=128,
                            )
                        ]
                    )
                ]
            )

            shifted = []
            for ds in downsampled:
                pil_down = Image.fromarray(ds)
                pil_down = rand_aff(pil_down)
                shifted.append(pil_down)

            downsampled = shifted

        ds_array = [np.array(ds, dtype=np.float32) for ds in downsampled]

        ds_array = [cv2.normalize(ds, ds, 0, 1, cv2.NORM_MINMAX) for ds in ds_array]

        if self.train and self.gaussian_noise:
            mu, std = self.gaussian_noise_settings
            ds_array = [
                ds
                + np.random.normal(
                    mu, std, size=(self.downsample_size, self.downsample_size)
                )
                for ds in ds_array
            ]

        ds_array = [torch.from_numpy(ds).float() for ds in ds_array]

        if len(ds_array) == 1:
            ds_array = ds_array[0].unsqueeze(0)
            ds_array = ds_array.repeat(3, 1, 1)
        else:
            ds_array = torch.stack(ds_array)

        labels = image_metadata[self.label_cols]
        labels = torch.tensor(labels)

        cont_labels = torch.tensor([])
        if len(self.cont_label_cols) > 0:
            cont_labels = image_metadata[self.cont_label_cols]
            cont_labels = torch.tensor(cont_labels)

        if self.include_demo:
            return ds_array, demo_tensor, labels, cont_labels
        else:
            return ds_array, labels, cont_labels


class StanfordCXRDataset(Dataset):

    def __init__(
        self,
        metadata_df_path,
        pat_identifier,
        label_cols,
        cont_label_cols=[],
        multi_view=False,
        include_demo=False,
        downsample_size=224,
        adaptive_norm=True,
        equalize_norm=False,
        brighten=False,
        gaussian_noise=True,
        gaussian_noise_settings=(0, 0.05),
        center_cropped=False,
        center_cropped_size=1400,
        multi_resolution=False,
        multi_resolution_sizes=["full", 1400, 900],
        rand_affine=False,
        age_stats=None,
        enc=None,
        train=False,
        handle_nans="replace_zero",
        handle_minus_one="replace_zero",
        filter_pa=False,
        filter_frontal=False,
        subset_N=sys.maxsize,
    ):

        self.metadata = pd.read_csv(metadata_df_path)

        if filter_frontal:
            self.metadata = self.metadata[self.metadata["Frontal/Lateral"] == "Frontal"]

        if filter_pa:
            self.metadata = self.metadata[self.metadata["AP/PA"] == "PA"]

        if subset_N:
            self.metadata = self.metadata[:subset_N]

        self.handle_nans = handle_nans
        self.handle_minus_one = handle_minus_one

        if self.handle_nans == "replace_zero":
            self.metadata[label_cols] = self.metadata[label_cols].fillna(0.0)
        elif self.handle_nans == "replace_ones":
            self.metadata[label_cols] = self.metadata[label_cols].fillna(1.0)
        elif self.handle_nans == "remove":
            self.metadata = self.metadata.dropna(subset=label_cols)

        if self.handle_minus_one == "replace_zero":
            self.metadata[label_cols] = self.metadata[label_cols].replace(-1.0, 0.0)
        elif self.handle_minus_one == "replace_ones":
            self.metadata[label_cols] = self.metadata[label_cols].replace(-1.0, 1.0)
        elif self.handle_minus_one == "remove":
            self.metadata = self.metadata[
                (self.metadata[label_cols] == -1.0).any(axis=1)
            ]

        self.label_cols = label_cols
        self.cont_label_cols = cont_label_cols
        # Unique deid for each row
        self.metadata["IMAGE_ID"] = np.unique(
            self.metadata[pat_identifier], return_inverse=True
        )[-1]

        self.mapping = self.metadata[["IMAGE_ID", pat_identifier]]

        self.metadata.set_index("IMAGE_ID", inplace=True)
        self.train = train
        self.multi_view = multi_view
        self.include_demo = include_demo
        self.downsample_size = downsample_size
        self.adaptive_norm = adaptive_norm
        self.equalize_norm = equalize_norm
        self.brighten = brighten
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_settings = gaussian_noise_settings
        self.center_cropped = center_cropped
        self.center_cropped_size = center_cropped_size
        self.multi_resolution = multi_resolution
        self.multi_resolution_sizes = multi_resolution_sizes
        self.rand_affine = rand_affine
        self.age_stats = age_stats
        self.enc = enc

        if self.include_demo:
            demo_vars = self.metadata[["Sex", "Age"]].to_numpy()

            if enc is None:
                enc = OneHotEncoder()
                self.encoded_sex = enc.fit_transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()
                self.enc = enc
            else:
                self.enc = enc
                self.encoded_sex = self.enc.transform(
                    demo_vars[:, 0].reshape(-1, 1)
                ).toarray()

            self.locs_to_ilocs = dict(
                zip(self.metadata.index, np.arange(len(self.metadata.index)))
            )

            if age_stats is None:
                self.age_mean = self.metadata["Age"].mean()
                self.age_var = self.metadata["Age"].var()
                self.metadata["Age"] = (
                    self.metadata["Age"] - self.metadata["Age"].mean()
                ) / np.sqrt(self.metadata["Age"].var())
                self.age_stats = (self.age_mean, self.age_var)
            else:
                self.metadata["Age"] = (self.metadata["Age"] - age_stats[0]) / np.sqrt(
                    age_stats[1]
                )

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        image_metadata = self.metadata.loc[item]

        if self.include_demo:
            demo_index = self.locs_to_ilocs[item]
            encoded_sex = self.encoded_sex[demo_index]
            encoded_age = image_metadata["Age"]
            demo_tensor = torch.from_numpy(
                np.concatenate((encoded_sex, np.array([encoded_age])))
            )
            demo_tensor = demo_tensor.float()

        path = image_metadata["Path"]

        try:
            image_array = [cv2.imread(path)[:, :, 0]]
            # image_array = dicom.read_file(path).pixel_array
            # image_array = [image_array]
        except:
            print("Error in reading DICOM file {}".format(path))
            return None

        if self.multi_resolution:
            image_array = Image.fromarray(image_array[0])
            multi_res = self.multi_resolution_sizes

            cropped_images = []
            for res in multi_res:
                if res == "full":
                    cropped_images.append(np.array(image_array))
                else:
                    center_crop = transforms.Compose([transforms.CenterCrop(res)])
                    image_array = center_crop(image_array)
                    cropped_images.append(np.array(image_array))

            image_array = cropped_images

        downsampled = [
            zoom_2D(im_arr, (self.downsample_size, self.downsample_size))
            for im_arr in image_array
        ]

        if self.center_cropped:
            image_array = Image.fromarray(downsampled[0])
            center_crop = transforms.Compose(
                [transforms.CenterCrop(self.center_cropped_size)]
            )
            image_array = center_crop(image_array)
            downsampled = [np.array(image_array)]

        if self.adaptive_norm:
            downsampled = [adaptive_normalization_param(ds) for ds in downsampled]

        if self.equalize_norm:
            downsampled = [equalize_normalization(ds) for ds in downsampled]

        if self.brighten:
            downsampled = [brighten(ds) for ds in downsampled]

        if self.train and self.rand_affine:
            rand_aff = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.RandomAffine(
                                degrees=0,
                                translate=(0.05, 0.05),
                                scale=(0.95, 1.05),
                                fillcolor=128,
                            )
                        ]
                    )
                ]
            )

            shifted = []
            for ds in downsampled:
                pil_down = Image.fromarray(ds)
                pil_down = rand_aff(pil_down)
                shifted.append(pil_down)

            downsampled = shifted

        ds_array = [np.array(ds, dtype=np.float32) for ds in downsampled]

        ds_array = [cv2.normalize(ds, ds, 0, 1, cv2.NORM_MINMAX) for ds in ds_array]

        if self.train and self.gaussian_noise:
            mu, std = self.gaussian_noise_settings
            ds_array = [
                ds
                + np.random.normal(
                    mu, std, size=(self.downsample_size, self.downsample_size)
                )
                for ds in ds_array
            ]

        ds_array = [torch.from_numpy(ds).float() for ds in ds_array]

        if len(ds_array) == 1:
            ds_array = ds_array[0].unsqueeze(0)
            ds_array = ds_array.repeat(3, 1, 1)
        else:
            ds_array = torch.stack(ds_array)

        labels = image_metadata[self.label_cols]
        labels = torch.tensor(labels)

        cont_labels = torch.tensor([])
        if len(self.cont_label_cols) > 0:
            cont_labels = image_metadata[self.cont_label_cols]
            cont_labels = torch.tensor(cont_labels)

        if self.include_demo:
            return ds_array, demo_tensor, labels, cont_labels
        else:
            return ds_array, labels, cont_labels


class CXRDatasetMultiView(Dataset):

    def __init__(self, metadata_df_path, pat_identifier, label_cols, train=False):

        self.metadata = pd.read_csv(metadata_df_path)
        self.label_cols = label_cols

        # Unique deid for each row
        self.metadata["IMAGES_ID"] = np.unique(
            self.metadata[pat_identifier], return_inverse=True
        )[-1]

        self.mapping = self.metadata[["IMAGES_ID", "filename"]]

        self.metadata.set_index("IMAGES_ID", inplace=True)
        self.train = train

    def __len__(self):
        return len(set(self.metadata.index))

    def __getitem__(self, item):

        images_metadata = self.metadata.loc[item]

        pa_image_path = images_metadata[images_metadata["ViewPosition"] == "PA"][
            "path"
        ].iloc[0]
        ll_image_path = images_metadata[images_metadata["ViewPosition"] == "LL"][
            "path"
        ].iloc[0]

        try:
            pa_image_array = dicom.read_file(pa_image_path).pixel_array
        except:
            print("Error in reading PA DICOM file {}".format(pa_image_path))
            return None

        try:
            ll_image_array = dicom.read_file(ll_image_path).pixel_array
        except:
            print("Error in reading LL DICOM file {}".format(ll_image_path))
            return None

        processed_images = [np.nan, np.nan]

        for i, image in enumerate([pa_image_array, ll_image_array]):

            downsampled = zoom_2D(image, (224, 224))
            downsampled = adaptive_normalization(downsampled)
            ds_array = np.array(downsampled, dtype=np.float32)
            ds_array = cv2.normalize(ds_array, ds_array, 0, 1, cv2.NORM_MINMAX)

            ds_array = torch.from_numpy(ds_array).float()
            ds_array = ds_array.unsqueeze(0)
            ds_array = ds_array.repeat(3, 1, 1)

            processed_images[i] = ds_array

        labels = images_metadata.iloc[0][self.label_cols]

        return processed_images[0], processed_images[1], torch.tensor(labels)


def collate_cxr(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def brighten(img):
    cols, rows = img.shape
    brightness = np.sum(img) / (255 * cols * rows)
    minimum_brightness = 0.6
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img
    else:
        return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


def adaptive_normalization(tensor, dim3d=False):
    """
    Contrast localized adaptive histogram normalization
    :param tensor: ndarray, 2d or 3d
    :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
    :return: normalized image
    """
    return hist.equalize_adapthist(tensor)  # , kernel_size=128, nbins=1024)


def adaptive_normalization_param(tensor, dim3d=False):
    """
    Contrast localized adaptive histogram normalization
    :param tensor: ndarray, 2'd or 3d
    :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
    :return: normalized image
    """
    return hist.equalize_adapthist(tensor, kernel_size=128, nbins=1024)


def equalize_normalization(tensor, dim3d=False):
    return hist.equalize_hist(tensor, nbins=256)


def zoom_2D(image, new_shape):
    """
    Uses open CV to resize a 2D image
    :param image: The input image, numpy array
    :param new_shape: New shape, tuple or array
    :return: the resized image
    """

    # OpenCV reverses X and Y axes
    return cv2.resize(
        image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC
    )


def convert_to_jpg(img_arr, equalize_before=True):
    min_p, max_p = img_arr.min(), img_arr.max()
    img_arr_jpg = (((img_arr - min_p) / max_p) * 255).astype(np.uint8)
    if equalize_before:
        img_arr_jpg = cv2.equalizeHist(img_arr_jpg)
    return img_arr_jpg
