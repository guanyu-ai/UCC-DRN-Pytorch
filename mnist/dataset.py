import os
import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import combinations
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
# from torch.utils.data import Dataset, DataLoader

root_dir:str = str(os.getenv("ROOT_DIR"))
data_dir = os.path.join(root_dir, "data/mnist/splitted_mnist_dataset.npz")
encoded_data_dir = os.path.join(root_dir, "data/mnist/splitted_mnist_encoded_dataset.npz")
drn_encoded_data_dir = os.path.join(root_dir, "data/mnist/splitted_drn_mnist_encoded_dataset.npz")encoded_data_dir = os.path.join(root_dir, "data/mnist/splitted_mnist_encoded_dataset.npz")


class MnistDataset(Dataset):
    def __init__(
        self,
        num_instances=2,
        num_samples_per_class=16,
        digit_arr=[],
        ucc_start=1,
        ucc_end=10,
        mode="train",
        length = 80000
    ):
        self._num_instances = num_instances # number of instances per bag
        self._num_samples_per_class = num_samples_per_class # number of bags per class in each batch
        self._digit_arr = digit_arr #  array of digits taken
        self._ucc_start = ucc_start # smallest ucc class
        self._ucc_end = ucc_end # largest ucc class
        self._num_digits = len(self._digit_arr) # number of digits in dataset, in this case is 10(0-9)
        self._num_classes = self._ucc_end - self._ucc_start + 1

        self.mode = mode
        self.length = length

        splitted_dataset = np.load(data_dir)

        # if self.mode =="train":
        x_train = splitted_dataset["x_train"]
        y_train = splitted_dataset["y_train"]
        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255
        x_train = (x_train - x_train.mean(dim=(2, 3), keepdim=True)) / x_train.std(
            dim=(2, 3), keepdim=True
        )

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")

        self._x_train = x_train
        self._y_train = torch.tensor(y_train, dtype=torch.int64)
        del x_train
        del y_train
        # elif self.mode =="val":
        x_val = splitted_dataset["x_val"]
        y_val = splitted_dataset["y_val"]
        x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1) / 255
        x_val = (x_val - x_val.mean(dim=(2, 3), keepdim=True)) / x_val.std(
            dim=(2, 3), keepdim=True
        )
        print(x_val.shape[0], "val samples")

        self._x_val = x_val
        self._y_val = torch.tensor(y_val, dtype=torch.int64)

        del x_val
        del y_val
        x_test = splitted_dataset['x_test']
        y_test = splitted_dataset['y_test']
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255
        x_test = (x_test - x_test.mean(dim=(2, 3), keepdim=True)) / x_test.std(
            dim=(2, 3), keepdim=True
        )
        self._x_test = x_test
        self._y_test = torch.tensor(y_test, dtype=torch.int64)
        x_test = splitted_dataset["x_test"]
        y_test = splitted_dataset["y_test"]
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255
        print(x_test.std())
        print(x_test.mean())
        x_test = (x_test - x_test.mean(dim=(2, 3), keepdim=True)) / x_test.std(
            dim=(2, 3), keepdim=True
        )
        
        print(x_test.shape[0], "test samples")
        self._x_test = x_test
        self._y_test = torch.tensor(y_test, dtype=torch.int64)
        del splitted_dataset

        # reshape to instances

        self._digit_dict = self.get_digit_dict()
        self._class_dict_train = self.get_class_dict()
        self._class_dict_val = self.get_class_dict()

        self._labels = self.generate_labels()

    def set_mode(self,mode:str):
        self.mode = mode

    def get_digit_dict(self):
        # digit_dict would look something like this
        # {
        #   "1": {
        #           "value": 1,
        #           "train_indices": [1,45,233,1111,34242,...],
        #           "num_train": 50000
        #       },
        #   ...
        # }
        # indexing get_digit_dict with ucc_class returns all indices of a specific index.
        digit_dict = dict()
        for i in range(self._num_digits):
            digit_key = "digit" + str(i)
            digit_value = self._digit_arr[i]

            temp_digit_dict = dict()

            temp_digit_dict["value"] = digit_value
            temp_digit_dict["train_indices"] = np.where(self._y_train == digit_value)[0]
            temp_digit_dict["num_train"] = len(temp_digit_dict["train_indices"])
            temp_digit_dict["val_indices"] = np.where(self._y_val == digit_value)[0]
            temp_digit_dict["num_val"] = len(temp_digit_dict["val_indices"])
            temp_digit_dict["test_indices"] = np.where(self._y_test == digit_value)[0]
            temp_digit_dict["num_test"] = len(temp_digit_dict["test_indices"])
            digit_dict[digit_key] = temp_digit_dict

        return digit_dict

    def get_class_dict(self):
        # class_dict would look something like this
        # {
        #   1: [[1],[2],[3]...],
        #   2: [[1,2], [2,3], [1,4], [4,6]...],
        #   3: [[1,2,3], [4,5,6]...],
        #   ...
        # }
        # indexing class_dict with ucc_class returns combinations of digits for each class.
        elements_arr = np.arange(self._num_digits)
        class_dict = dict()

        for i in range(self._num_classes):  # for each ucc class
            class_key = "class_" + str(i)
            elements_list = list()
            for j in combinations(elements_arr, i + self._ucc_start):   # get combinations of instance classes
                elements_list.append(np.array(j))
            elements_array = np.array(elements_list)
            np.random.shuffle(elements_array)
            class_dict[class_key] = np.array(elements_array)
        return class_dict

    def one_hot_label(self, label):
        one_hot_label = np.zeros(self._num_classes, dtype=np.int32)
        one_hot_label[label] = 1
        return one_hot_label

    def generate_labels(self):
        labels_list = list()
        for i in range(self._num_classes):
            labels_list.append(self.one_hot_label(i))

        labels_arr = np.repeat(
            np.array(labels_list), self._num_samples_per_class, axis=0
        )
        # print(labels_arr)

        return labels_arr

    def get_sample_data_train(self, indices_arr):
        sample = np.array(self._x_train[indices_arr, :, :, :])
        return sample

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # if self.mode = ""
        # get one bag of a lot of instances of images
        # shape should be [instance_per_class, *[image dimensions]]
        class_label = index%self._num_classes   # get a class label
        class_key = f"class_{class_label}"
        ind = np.random.randint(0, len(self._class_dict_train[class_key]))
        elems = self._class_dict_train[class_key][ind, :]
        num_elems = len(elems)
        num_instances_per_elem = self._num_instances//num_elems
        remainder = self._num_instances%num_elems
        num_instances_arr = np.repeat(num_instances_per_elem, num_elems)
        num_instances_arr[:remainder]+=1

        indices_list = []
        for k in range(num_elems):
            digit_key = f"digit{elems[k]}"  # get digit labels
            num_instances = num_instances_arr[k]
            train_size = len(self._digit_dict[digit_key]["train_indices"])
            val_size = len(self._digit_dict[digit_key]["val_indices"])
            test_size = len(self._digit_dict[digit_key]["test_indices"])

            if self.mode=="train":
                random_ind = np.random.randint(
                    0, train_size, num_instances
                )
                indices_list += list(
                    self._digit_dict[digit_key]["train_indices"][random_ind]
                )
                samples = self._x_train[indices_list]

            elif self.mode=="val":
                random_ind = np.random.randint(
                    0, val_size, num_instances
                )
                indices_list += list(
                    self._digit_dict[digit_key]["val_indices"][random_ind]
                )
                samples = self._x_val[indices_list]
            elif self.mode=="test":
                random_ind = np.random.randint(
                    0, test_size, num_instances
                )
                indices_list += list(
                    self._digit_dict[digit_key]["test_indices"][random_ind]
                )
                samples = self._x_test[indices_list]
        samples = samples.view(
            self._num_instances, 1, self._x_train.shape[2], self._x_train.shape[3]
        )
        return samples, class_label

    def get_sample_data_val(self, indices_arr):
        sample = np.array(self._x_val[indices_arr, :, :, :])

        return sample



class MnistEncodedDataset(MnistDataset):
    def __init__(self, num_instances=2, num_samples_per_class=16, digit_arr=[], ucc_start=1, ucc_end=10, mode="train", length=80000):
        super().__init__(num_instances, num_samples_per_class, digit_arr, ucc_start, ucc_end, mode, length)
        self._num_instances = num_instances # number of instances per bag
        self._num_samples_per_class = num_samples_per_class # number of bags per class in each batch
        self._digit_arr = digit_arr #  array of digits taken
        self._ucc_start = ucc_start # smallest ucc class
        self._ucc_end = ucc_end # largest ucc class
        self._num_digits = len(self._digit_arr) # number of digits in dataset, in this case is 10(0-9)
        self._num_classes = self._ucc_end - self._ucc_start + 1

        self.mode = mode
        self.length = length

        splitted_dataset = np.load(encoded_data_dir)

        # if self.mode =="train":
        x_train = splitted_dataset["x_train"]
        y_train = splitted_dataset["y_train"]
        x_train = torch.tensor(x_train, dtype=torch.float32)

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")

        self._x_train = x_train
        self._y_train = torch.tensor(y_train, dtype=torch.int64)
        del x_train
        del y_train
        # elif self.mode =="val":
        x_val = splitted_dataset["x_val"]
        y_val = splitted_dataset["y_val"]
        x_val = torch.tensor(x_val, dtype=torch.float32)

        print(x_val.shape[0], "val samples")

        self._x_val = x_val
        self._y_val = torch.tensor(y_val, dtype=torch.int64)

        del x_val
        del y_val
        # x_test = splitted_dataset['x_test']
        # y_test = splitted_dataset['y_test']

        del splitted_dataset

        # reshape to instances

        self._digit_dict = self.get_digit_dict()
        self._class_dict_train = self.get_class_dict()
        self._class_dict_val = self.get_class_dict()

        self._labels = self.generate_labels()

    def __getitem__(self, index):
        # if self.mode = ""
        # get one bag of a lot of instances of images
        # shape should be [instance_per_class, *[image dimensions]]
        class_label = index%self._num_classes   # get a class label
        class_key = f"class_{class_label}"
        ind = np.random.randint(0, len(self._class_dict_train[class_key]))
        elems = self._class_dict_train[class_key][ind, :]
        num_elems = len(elems)
        num_instances_per_elem = self._num_instances//num_elems
        remainder = self._num_instances%num_elems
        num_instances_arr = np.repeat(num_instances_per_elem, num_elems)
        num_instances_arr[:remainder]+=1

        indices_list = []
        for k in range(num_elems):
            digit_key = f"digit{elems[k]}"  # get digit labels
            num_instances = num_instances_arr[k]
            train_size = len(self._digit_dict[digit_key]["train_indices"])
            val_size = len(self._digit_dict[digit_key]["val_indices"])
            random_ind = np.random.randint(
                0, train_size if self.mode=="train" else val_size, num_instances
            )
            if self.mode=="train":
                random_ind = np.random.randint(
                    0, train_size, num_instances
                )
                indices_list += list(
                    self._digit_dict[digit_key]["train_indices"][random_ind]
                )
                samples = self._x_train[indices_list]

            elif self.mode=="val":
                random_ind = np.random.randint(
                    0, val_size, num_instances
                )
                indices_list += list(
                    self._digit_dict[digit_key]["val_indices"][random_ind]
                )
                samples = self._x_val[indices_list]
        return samples, class_label
    
class DRNMnistEncodedDataset(MnistDataset):
    def __init__(self, num_instances=2, num_samples_per_class=16, digit_arr=[], ucc_start=1, ucc_end=10, mode="train", length=80000):
        super().__init__(num_instances, num_samples_per_class, digit_arr, ucc_start, ucc_end, mode, length)
        self._num_instances = num_instances # number of instances per bag
        self._num_samples_per_class = num_samples_per_class # number of bags per class in each batch
        self._digit_arr = digit_arr #  array of digits taken
        self._ucc_start = ucc_start # smallest ucc class
        self._ucc_end = ucc_end # largest ucc class
        self._num_digits = len(self._digit_arr) # number of digits in dataset, in this case is 10(0-9)
        self._num_classes = self._ucc_end - self._ucc_start + 1

        self.mode = mode
        self.length = length

        splitted_dataset = np.load(drn_encoded_data_dir)
        # if self.mode =="train":
        x_train = splitted_dataset["x_train"]
        y_train = splitted_dataset["y_train"]
        x_train = torch.tensor(x_train, dtype=torch.float32)

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")

        self._x_train = x_train
        self._y_train = torch.tensor(y_train, dtype=torch.int64)
        del x_train
        del y_train
        # elif self.mode =="val":
        x_val = splitted_dataset["x_val"]
        y_val = splitted_dataset["y_val"]
        x_val = torch.tensor(x_val, dtype=torch.float32)

        print(x_val.shape[0], "val samples")

        self._x_val = x_val
        self._y_val = torch.tensor(y_val, dtype=torch.int64)

        del x_val
        del y_val
        # x_test = splitted_dataset['x_test']
        # y_test = splitted_dataset['y_test']

        del splitted_dataset

        # reshape to instances

        self._digit_dict = self.get_digit_dict()
        self._class_dict_train = self.get_class_dict()
        self._class_dict_val = self.get_class_dict()

        self._labels = self.generate_labels()

    def __getitem__(self, index):
        # if self.mode = ""
        # get one bag of a lot of instances of images
        # shape should be [instance_per_class, *[image dimensions]]
        class_label = index%self._num_classes   # get a class label
        class_key = f"class_{class_label}"
        ind = np.random.randint(0, len(self._class_dict_train[class_key]))
        elems = self._class_dict_train[class_key][ind, :]
        num_elems = len(elems)
        num_instances_per_elem = self._num_instances//num_elems
        remainder = self._num_instances%num_elems
        num_instances_arr = np.repeat(num_instances_per_elem, num_elems)
        num_instances_arr[:remainder]+=1

        indices_list = []
        for k in range(num_elems):
            digit_key = f"digit{elems[k]}"  # get digit labels
            num_instances = num_instances_arr[k]
            train_size = len(self._digit_dict[digit_key]["train_indices"])
            val_size = len(self._digit_dict[digit_key]["val_indices"])
            random_ind = np.random.randint(
                0, train_size if self.mode=="train" else val_size, num_instances
            )
            if self.mode=="train":
                indices_list += list(
                    self._digit_dict[digit_key]["train_indices"][random_ind]
                )
                samples = self._x_train[indices_list]

            elif self.mode=="val":
                indices_list += list(
                    self._digit_dict[digit_key]["val_indices"][random_ind]
                )
                samples = self._x_val[indices_list]
        return samples, class_label
