import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchsample.transforms as ts
from tqdm import tqdm
from dataio.loader.ct_82_dataset import CT82Dataset
from dataio.loader.utils import get_dicom_dirs, get_train_test_val_indices
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models import get_model
from torch.utils.tensorboard import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(arguments):

    # Parse input arguments
    json_filename = arguments.config

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Create train-test-validation splits
    np.random.seed(28)
    root_dir = json_opts.data_path.ct_82
    num_files = len(get_dicom_dirs(os.path.join(root_dir, "image")))
    train_idx, test_idx, val_idx = get_train_test_val_indices(num_files, test_frac=0.25, val_frac=0.0)

    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)
    train_dataset = CT82Dataset(root_dir, "train", train_idx, transform=ds_transform['train'], resample=True, preload_data=train_opts.preloadData)
    test_dataset = CT82Dataset(root_dir, "test", test_idx, transform=ds_transform['valid'], resample=True, preload_data=train_opts.preloadData)
    # val_dataset = CT82Dataset(root_dir, "validation", val_idx, transform=ds_transform['valid'], resample=True)

    # Setup the NN Model
    model = get_model(json_opts.model)

    # Setup Data Loaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,  num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)

    # Define tensorboard writer
    writer = SummaryWriter(os.path.join(model.save_dir, "runs"))

    ## Add model architecture to TensorBoard
    # images, labels = iter(train_loader).next()
    # images = images.to('cuda') if model.use_cuda else images
    # writer.add_graph(model.net, images)
    # writer.flush()

    # Training
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        stats_dict = {}
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            for error_name, error in errors.items():
                if 'Train ' + error_name in stats_dict:
                    stats_dict['Train ' + error_name].append(error)
                else:
                    stats_dict['Train ' + error_name] = [error]

        # Validation and Testing Iterations
        with torch.no_grad():
            for epoch_iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                errors = model.get_current_errors()
                for error_name, error in errors.items():
                    if 'Train ' + error_name in stats_dict:
                        stats_dict['Train ' + error_name].append(error)
                    else:
                        stats_dict['Train ' + error_name] = [error]
                stats = model.get_segmentation_stats()
                for stat_name, stat in stats.items():
                    if 'Test ' + stat_name in stats_dict:
                        stats_dict['Test ' + stat_name].append(stat)
                    else:
                        stats_dict['Test ' + stat_name] = [stat]

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()

        for k, v in stats_dict.items():
            writer.add_scalar(k, sum(v)/len(v), epoch)
        writer.flush()

    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    args = parser.parse_args()

    train(args)
