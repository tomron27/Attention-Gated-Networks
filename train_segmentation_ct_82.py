import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataio.loader.ct_82_dataset import CT82Dataset
from dataio.loader.utils import get_dicom_dirs, get_train_test_val_indices
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    # ds_class = get_dataset(arch_type)
    # ds_path = get_dataset_path(arch_type, json_opts.data_path)
    # ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Create train-test-validation splits
    np.random.seed(42)
    root_dir = "/home/tomron27/datasets/CT-82/"
    num_files = len(get_dicom_dirs(os.path.join(root_dir, "image")))
    train_idx, test_idx, val_idx = get_train_test_val_indices(num_files, test_frac=0.25, val_frac=0.0)

    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)
    train_dataset = CT82Dataset(root_dir, "train", train_idx, transform=ds_transform['train'], resample=True)
    test_dataset = CT82Dataset(root_dir, "test", test_idx, transform=ds_transform['valid'], resample=True)
    # val_dataset = CT82Dataset(root_dir, "validation", val_idx, transform=ds_transform['valid'], resample=True)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,  num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        with torch.no_grad():
            for epoch_iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split='test')

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

            # Update the plots
            for split in ['train', 'test']:
                visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
                visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
