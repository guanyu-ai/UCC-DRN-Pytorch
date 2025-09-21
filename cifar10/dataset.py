import os
import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import combinations
import dotenv
# dotenv.load_dotenv(dotenv.find_dotenv())
# from torch.utils.data import Dataset, DataLoader
# root_dir:str = str(os.getenv("ROOT_DIR"))
data_dir = "../data/cifar10/splitted_cifar10_dataset.npz"

class Cifar10Dataset(Dataset):
    def __init__(
         self,
        num_instances=2,
        object_arr=[],
        ucc_start=1,
        ucc_end=10,
        mode="train",
        length = 80000
    ):
        self._num_instances = num_instances # number of instances per bag
        self._object_arr = object_arr #  array of digits taken
        self._ucc_start = ucc_start # smallest ucc class
        self._ucc_end = ucc_end # largest ucc class
        self._num_objects = len(self._object_arr) # number of objects in dataset, in this case is 10(0-9)
        self._num_classes = self._ucc_end - self._ucc_start + 1

        self.mode = mode
        self.length = length

        # TODO Later check if the shape of the input image is correct or not, and also the normalization methods etc.
        splitted_dataset = np.load(data_dir)
        # load data
        # shape: (num_samples, 512, 512, 3)
        x_train = splitted_dataset['x_train']
        y_train = splitted_dataset['y_train']
        x_train = torch.tensor(x_train, dtype=torch.float32)/ 255
        x_train = (x_train - x_train.mean(dim=(0, 1, 2), keepdim=True)) / x_train.std(
            dim=(0, 1, 2), keepdim=True
        )
        x_train = x_train.permute(0,3,1,2)
        self._x_train = x_train
        self._y_train = torch.tensor(y_train, dtype=torch.int64)
        del x_train
        del y_train
  
        x_val = splitted_dataset["x_val"]
        y_val = splitted_dataset["y_val"]
        x_val = torch.tensor(x_val, dtype=torch.float32)/255
        x_val = (x_val - x_val.mean(dim=(0, 1, 2), keepdim=True)) / x_val.std(
            dim=(0, 1, 2), keepdim=True
        )
        x_val = x_val.permute(0,3,1,2)
        print(x_val.shape[0], "val samples")
        self._x_val = x_val
        self._y_val = torch.tensor(y_val, dtype=torch.int64)
        del x_val
        del y_val
        
        x_test = splitted_dataset['x_test']
        y_test = splitted_dataset['y_test']
        x_test = torch.tensor(x_test, dtype=torch.float32)/255
        x_test = (x_test - x_test.mean(dim=(0, 1, 2), keepdim=True)) / x_test.std(
            dim=(0, 1, 2), keepdim=True
        )
        x_test = x_test.permute(0,3,1,2)
        
        self._x_test = x_test
        self._y_test = torch.tensor(y_test, dtype=torch.int64)
        
        del x_test
        del y_test
        self._object_dict = self.get_object_dict()
        self._class_dict_train = self.get_class_dict()
        # self._class_dict_val = self.get_class_dict()


    def get_object_dict(self):
        object_dict = dict()
		# for object every combination
        for i in range(self._num_objects):
            object_key = 'object' + str(i)
            object_value = self._object_arr[i]

            temp_object_dict = dict()

            temp_object_dict['value'] = object_value
            temp_object_dict['train_indices'] = np.where(self._y_train == object_value)[0]
            temp_object_dict['num_train'] = len(temp_object_dict['train_indices'])
            temp_object_dict['val_indices'] = np.where(self._y_val == object_value)[0]
            temp_object_dict['num_val'] = len(temp_object_dict['val_indices'])
            temp_object_dict["test_indices"] = np.where(self._y_test == object_value)[0]
            temp_object_dict["num_test"] = len(temp_object_dict["test_indices"])

            # print('{}:{}, num_train:{}, num_val:{}'.format(object_key,object_value,temp_object_dict['num_train'],temp_object_dict['num_val']))

            object_dict[object_key] = temp_object_dict

        return object_dict
    
    def get_class_dict(self):
		# for object every combination
        elements_arr = np.arange(self._num_objects)
        class_dict = dict()
        for i in range(self._num_classes):
            class_key = 'class_' + str(i)
            # print(elements_arr)
            elements_list = list()
            for j in combinations(elements_arr,i+self._ucc_start):
                elements_list.append(np.array(j))
            elements_array = np.array(elements_list)
            np.random.shuffle(elements_array)
            class_dict[class_key] = elements_array
            
        return class_dict
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
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
            object_key = f"object{elems[k]}"  # get digit labels
            num_instances = num_instances_arr[k]
            train_size = len(self._object_dict[object_key]["train_indices"])
            val_size = len(self._object_dict[object_key]["val_indices"])
            test_size = len(self._object_dict[object_key]["test_indices"])

            if self.mode=="train":
                random_ind = np.random.randint(
                    0, train_size, num_instances
                )
                indices_list += list(
                    self._object_dict[object_key]["train_indices"][random_ind]
                )
                samples = self._x_train[indices_list]

            elif self.mode=="val":
                random_ind = np.random.randint(
                    0, val_size, num_instances
                )
                indices_list += list(
                    self._object_dict[object_key]["val_indices"][random_ind]
                )
                samples = self._x_val[indices_list]
            elif self.mode=="test":
                random_ind = np.random.randint(
                    0, test_size, num_instances
                )
                indices_list += list(
                    self._object_dict[object_key]["test_indices"][random_ind]
                )
                samples = self._x_test[indices_list]
        samples = samples.view(
            self._num_instances, 3, self._x_train.shape[2], self._x_train.shape[3]
        )
        return samples, class_label

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


    def get_sample_data_train(self, indices_arr):
        sample = np.array(self._x_train[indices_arr, :, :, :])
        return sample
    
    def get_sample_data_val(self, indices_arr):
        sample = np.array(self._x_val[indices_arr, :, :, :])

        return sample
        