import os
import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import combinations
import dotenv
# dotenv.load_dotenv(dotenv.find_dotenv())
# from torch.utils.data import Dataset, DataLoader
# root_dir:str = str(os.getenv("ROOT_DIR"))
dataset_dir = "../data/camelyon"

class CamelyonDataset(Dataset):
    def __init__(
        self,
        mode="train",
        num_instances=10,
        data_augment=False,
        patch_size=None,
        dataset_len=1000,
    ):
        self.mode = mode
        # print(f"{dataset_dir}/{mode}_img_data.npy")
        self.num_instances = num_instances
        self.data_augment = data_augment
        self.patch_size = patch_size
        self.dataset_len = dataset_len

        # load data
        # shape: (num_samples, 512, 512, 3)
        self.x = np.load(f"{dataset_dir}/{mode}_img_data.npy")
        self.y = np.load(f"{dataset_dir}/{mode}_ucc_label.npy")
        self.y_mask = np.load(f"{dataset_dir}/{mode}_mask_data.npy")

        self.image_size = self.x.shape[1]
        self.range_high_lim = self.image_size - self.patch_size + 1

        self._sample_indices_ucc1 = np.where(self.y[:,0]==1)[0]
        self._sample_indices_ucc2 = np.where(self.y[:,1]==1)[0]
        self.num_samples = self.y.shape[0]
        self.num_classes = self.y.shape[1]

    def __len__(self):
        return self.dataset_len if self.mode != "test" else self.num_samples

    def __getitem__(self, index):
        ucc_class = index % self.num_classes
        # shape: (num_instances, patch_size, patch_size, 3)

        if ucc_class == 0:
            index = np.random.choice(self._sample_indices_ucc1)
        elif ucc_class == 1:
            index = np.random.choice(self._sample_indices_ucc2)
        sample_data = self.get_sample_data(index)
        # normalize height and width dimension
        sample_data = (
            sample_data
            - np.mean(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        ) / np.std(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        
        sample_data = np.asarray(sample_data, dtype=np.float32)
        sample_data = torch.from_numpy(sample_data).permute(0, 3, 1, 2)
        label = torch.from_numpy(self.y[index]).long()
        # convert label to class index
        label = torch.argmax(label)
        
        return sample_data, label

    def augment_image(self, image, augment_ind):
        if augment_ind == 0:
            return image
        elif augment_ind == 1:
            return np.rot90(image)
        elif augment_ind == 2:
            return np.rot90(image, 2)
        elif augment_ind == 3:
            return np.rot90(image, 3)
        elif augment_ind == 4:
            return np.fliplr(image)
        elif augment_ind == 5:
            return np.rot90(np.fliplr(image))
        elif augment_ind == 6:
            return np.rot90(np.fliplr(image), 2)
        elif augment_ind == 7:
            return np.rot90(np.fliplr(image), 3)


    def get_sample_data(self, sample_ind):
        image_arr = self.x[sample_ind]

        instance_list = list()
        for _ in range(self.num_instances):
            r, c = np.random.randint(low=0, high=self.range_high_lim, size=2)

            patch_data = image_arr[r : r + self.patch_size, c : c + self.patch_size, :]

            if self.data_augment:
                augment_id = np.random.randint(8)
                patch_data = self.augment_image(patch_data, augment_id)

            instance_list.append(patch_data)

        instance_arr = np.array(instance_list)

        return instance_arr

    def get_image_patches(self, sample_ind):
        # get all patches from whole image
        image_arr = self.x[sample_ind]
        image_size = image_arr.shape[0]
        patch_list = list()
        for r in range(0, image_size, self.patch_size):
            for c in range(0, image_size, self.patch_size):
                patch_list.append(
                    image_arr[r : r + self.patch_size, c : c + self.patch_size, :]
                )
        patches = np.array(patch_list, dtype=np.float32)
        patches = (
            patches - np.mean(patches, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        ) / np.std(patches, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        patches = torch.from_numpy(patches)
        # reshape patches to (num_patches, num_channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 2)
        return patches

    def get_image_patch_labels(self, sample_ind):
        # must convert pixel value to 0-1 not 0-255
        image_mask = self.y_mask[sample_ind] / 255
        image_size = image_mask.shape[0]
        patch_truth_list = list()
        for r in range(0, image_size, self.patch_size):
            for c in range(0, image_size, self.patch_size):
                mask_patch = image_mask[
                    r : r + self.patch_size, c : c + self.patch_size
                ]
                metastasis_ratio = np.sum(mask_patch) / (
                    self.patch_size * self.patch_size
                )

                if metastasis_ratio > 0.5:
                    patch_truth_list.append(1)
                else:
                    patch_truth_list.append(0)
        return np.asarray(patch_truth_list)
        
        
class CamelyonDatasetSeparatedBin(CamelyonDataset):
    def __getitem__(self, index):
        ucc_class = index % self.num_classes
        # shape: (num_instances, patch_size, patch_size, 3)

        if ucc_class == 0:
            index = np.random.choice(self._sample_indices_ucc1)
        elif ucc_class == 1:
            index = np.random.choice(self._sample_indices_ucc2)
        sample_data = self.get_sample_data(index)
        # normalize height and width dimension
        sample_data = (
            sample_data
            - np.mean(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        ) / np.std(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        
        sample_data = np.asarray(sample_data, dtype=np.float32)
        sample_data = torch.from_numpy(sample_data).permute(0, 3, 1, 2)
        label = torch.from_numpy(self.y[index]).long()
        # convert label to class index
        label = torch.argmax(label)
        if label == 0:
            label = torch.tensor([1,0,0,0], dtype=torch.float32)
        elif label == 1:
            label = torch.tensor([0,0,0,1], dtype=torch.float32)
        return sample_data, label