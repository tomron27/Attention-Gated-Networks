import torch
import torch.utils.data as data
import h5py
import numpy as np
import datetime
import os

from os import listdir
from os.path import join
from scipy.misc import imresize
#from .utils import check_exceptions


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

class UltraSoundDataset1(data.Dataset):
    def __init__(self, root_path, split, transform=None, preload_data=False):
        super(UltraSoundDataset1, self).__init__()

        self.n_class = 2
        data_files = ["NORM_all_224x288_RVOT.hdf5", 'NORM_all_224x288_4CH.hdf5', "HLH_all_224x288_RVOT.hdf5", 'HLH_all_224x288_4CH.hdf5' ]
        data_labels = [0, 0, 1, 1]
        #data_files = ["NORM_all_224x288_RVOT.hdf5", "HLH_all_224x288_RVOT.hdf5"]
        #data_labels = [0, 1]
        #TODO all images are in images_test
        if split == 'train':
            imagekey = 'images_train'
            labelkey = 'plane_labels_train'
        else: 
            imagekey = 'images_test'
            labelkey = 'plane_labels_test'

        tmpimg = []
        tmplbls = []
        pimgs = []
        plabels = []
        self.data = []
        self.targets = []

        labelnames = []
        labelnumbers = []
        print(os.getcwd())

        for h in range(len(data_files)):
            print('loading %s' % data_files[h])
            h5py_dict = h5py.File(data_files[h], 'r')
            self.data.append(h5py_dict.get(imagekey)[()])
            targets = np.array(h5py_dict.get(labelkey)[()])
            targets.fill(data_labels[h])
            self.targets.append(targets)
            print(self.data[h].shape)
            print(self.targets[h])


        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.data, self.targets = unison_shuffled_copies(self.data, self.targets)
        self.images = self.data #torch.from_numpy(self.data).float()
        self.labels = self.targets #torch.from_numpy(self.targets).long()
        self.label_names = ['healthy', 'pathologic']

        class_weight = np.zeros(self.n_class)
        for lab in range(self.n_class):
            class_weight[lab] = np.sum(self.labels[:] == lab)

        class_weight = 1 / class_weight
    
        self.weight = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            self.weight[i] = class_weight[self.labels[i]]

        #print(class_weight)
        assert len(self.images) == len(self.labels)

        # data augmentation
        self.transform = transform

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        img  = self.images[index][0]
        target = self.labels[index]

        oshape = img.shape
        img = np.nan_to_num(img)
        img = (img-img.mean())/img.var(); 
        if (img.max() - img.min()) != 0:
            img = (img - img.min())/(img.max() - img.min())

        img = np.nan_to_num(img)
        img = img[50:img.shape[0]-30, 20:img.shape[1]-50]
        img = autocrop(img, 0.05)
        img = imresize(img, oshape)

        #input = input.transpose((1,2,0))

        # handle exceptions
        #check_exceptions(input, target)
        if self.transform:
            img = self.transform(img)

        img = np.nan_to_num(img)

        #print(input.shape, torch.from_numpy(np.array([target])))
        #print("target",np.int64(target))
        return img, int(target)

    def __len__(self):
        return len(self.images)


# if __name__ == '__main__':
#     dataset = UltraSoundDataset("/vol/bitbucket/js3611/data_ultrasound/preproc_combined_inp_224x288.hdf5",'test')

#     from torch.utils.data import DataLoader, sampler
#     ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
