import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz", ".dcm"])


def get_train_test_val_indices(n, test_frac=0.3, val_frac=0.5):
    """
    Creates a train-test-validation split out of given index range
    :param n: index range
    :param test_frac: fraction for test AND validation set
    :param val_frac: fraction for validation set OUT OF test set
    :return: train, test, val indices
    """
    indices = np.array(range(n))
    np.random.shuffle(indices)
    train_split = int(np.floor(n*(1-test_frac)))
    test_split = int(np.floor(n*(1-test_frac*val_frac)))
    train = indices[:train_split]
    test = indices[train_split:test_split]
    val = indices[test_split:]
    return train, test, val


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
    return img.copy()


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
    return mask.copy()


def get_dicom_dirs(parent_folder):
    """
    Get most-down directories of all dicom sequences under parent folder
    :param parent_folder: parent folder of image set
    :return: List of most-down directories containing .dcm sequences
    """
    res = []
    for dirpath, dirnames, filenames in os.walk(parent_folder):
        if not dirnames and len(filenames) > 0 and is_image_file(filenames[0]):
            res.append(dirpath)
    return res


def get_nifti_files(parent_folder):
    """
    Get all NifTi files under parent folder
    :param parent_folder: parent folder of NifTi files
    :return: List of NifTi files
    """
    res = []
    for dirpath, dirnames, filenames in os.walk(parent_folder):
        for file in filenames:
            if file.endswith(".nii.gz"):
                res.append(os.path.join(dirpath, file))
    return res


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
    os.makedirs(savedir, exist_ok=True)
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


if __name__ == "__main__":
    test_path = "/home/tomron27/datasets/CT-82/image/"
    l = get_dicom_dirs(test_path)
    print(l)
