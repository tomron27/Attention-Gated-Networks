import torch
# from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import datetime

from os.path import join
from .utils import check_exceptions, get_dicom_dirs, get_nifti_files, load_dicom_dir, load_nifti_mask


class CT82Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, split_indices, transform=None, preload_data=False):
        super(CT82Dataset, self).__init__()

        # Load image / label files
        self.image_filenames = sorted(get_dicom_dirs(join(root_dir, 'image')))
        self.target_filenames = sorted(get_nifti_files(join(root_dir, 'label')))

        # Filter according to split indices (train / test / val)
        self.image_filenames = np.array(self.image_filenames)[split_indices].tolist()
        self.target_filenames = np.array(self.target_filenames)[split_indices].tolist()

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Split: {0}, Total number of images: {1} Dicoms'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the dataset ...')
            self.raw_images = [load_dicom_dir(ii) for ii in self.image_filenames]
            self.raw_labels = [load_nifti_mask(ii) for ii in self.target_filenames]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input = load_dicom_dir(self.image_filenames[index])
            target = load_nifti_mask(self.target_filenames[index])
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        check_exceptions(input, target)
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)