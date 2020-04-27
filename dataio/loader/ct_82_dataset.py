import torch
# from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import datetime

from os.path import join
from .utils import check_exceptions, get_dicom_dirs, get_nifti_files, load_image_and_mask


class CT82Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, split_indexes, transform=None, preload_data=False, resample=False):
        super(CT82Dataset, self).__init__()

        self.split_indexes = split_indexes
        # Load image / label files
        self.image_filenames = [image for i, image in enumerate(sorted(get_dicom_dirs(join(root_dir, 'image')))) if i in split_indexes]
        self.target_filenames = [image for i, image in enumerate(sorted(get_nifti_files(join(root_dir, 'label')))) if i in split_indexes]

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Split: {0}, Total number of images: {1} Dicoms'.format(split, len(self.image_filenames)))

        # data augmentation
        self.transform = transform

        # Size resampling
        self.resample = resample

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the dataset ...')
            data = [load_image_and_mask(ii, self.image_filenames, self.target_filenames, self.resample)
                    for ii in split_indexes]
            self.raw_images,  self.raw_labels = [[i for i, j in data], [j for i, j in data]]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the dicom + nifti mask
        if not self.preload_data:
            load_index = self.split_indexes[index]
            input, target = load_image_and_mask(load_index, self.image_filenames, self.target_filenames, self.resample)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        check_exceptions(input, target, self.image_filenames[index])
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
