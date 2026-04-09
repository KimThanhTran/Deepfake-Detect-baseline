import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import os
import sys
from .datasets import dataset_folder


def get_dataset(opt):
    if not os.path.isdir(opt.dataroot):
        raise FileNotFoundError(
            f"Dataset path not found: {opt.dataroot}. "
            "Expected a dataset root or split directory containing class folders."
        )
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    if '0_real' not in classes or '1_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = opt.dataroot + '/' + cls
            if not os.path.isdir(root):
                raise FileNotFoundError(
                    f"Class directory not found: {root}. "
                    f"Available entries under {opt.dataroot}: {classes}"
                )
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, opt.dataroot)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    # Use 0 num_workers on Windows to avoid multiprocessing issues with lambda functions
    num_workers = 0 if sys.platform == 'win32' else int(opt.num_threads)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers)
    return data_loader
