import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.ct_82_dataset import CT82Dataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.us_dataset1 import UltraSoundDataset1


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax':  HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'us1': UltraSoundDataset1,
        'ct_82': CT82Dataset,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
