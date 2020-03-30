import nibabel as nib
import numpy as np
import os
from utils.util import mkdir
import SimpleITK as sitk


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz", ".dcm"])


def load_dicom_dir(dicom_path):
    """
    Load a folder with 3D dicom sequence (for CT-82 Dataset)
    :param dicom_path: str, path to dicom folder
    :return: ndarray, 3D array representation of volumetric data
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    img = sitk.GetArrayFromImage(img)[::-1]
    img = np.flip(img, axis=0)
    img = np.rot90(img, k=2, axes=(1,2))
    return img


def load_nifti_mask(mask_path):
    """
    Load NifTi 3D mask data, to work in conjunction with load_dicom_dir (for CT-82 Dataset)
    :param mask_path: path to "labelXXXX.nii.gz" mask file
    :return: ndarray, 3D array representation of volumetric mask data
    """
    mask = nib.load(mask_path)
    mask = np.transpose(mask.get_fdata(), (2, 0, 1))
    mask = np.flip(mask, axis=1)
    mask = np.rot90(mask, k=1, axes=(1,2))
    return mask


def load_nifti_img(filepath, dtype):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta


def write_nifti_img(input_nii_array, meta, savedir):
    mkdir(savedir)
    affine = meta['affine'][0].cpu().numpy()
    pixdim = meta['pixdim'][0].cpu().numpy()
    dim    = meta['dim'][0].cpu().numpy()

    img = nib.Nifti1Image(input_nii_array, affine=affine)
    img.header['dim'] = dim
    img.header['pixdim'] = pixdim

    savename = os.path.join(savedir, meta['name'][0])
    print('saving: ', savename)
    nib.save(img, savename)


def check_exceptions(image, label=None):
    if label is not None:
        if image.shape != label.shape:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
            #print('Skip {0}, {1}'.format(image_name, label_name))
            raise(Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank image exception'))