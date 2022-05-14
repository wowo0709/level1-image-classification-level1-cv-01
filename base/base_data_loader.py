import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    ### Initialization of BaseDataLoader
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        # Whether to split validation dataset
        self.validation_split = validation_split
        # Wheter to shuffle data
        self.shuffle = shuffle

        # batch index
        self.batch_idx = 0
        # total num of samples (length of data)
        self.n_samples = len(dataset)

        # sampler and valid sampler
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # keyword arguments
        self.init_kwargs = {
            'dataset': dataset, 
            'batch_size': batch_size, 
            'shuffle': self.shuffle, 
            'collate_fn': collate_fn, 
            'num_workers': num_workers
        }
        # call super()
        super().__init__(sampler=self.sampler, **self.init_kwargs)


    ### 
    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        # whole index list of data
        idx_full = np.arange(self.n_samples)

        # shuffle indices
        np.random.seed(0)
        np.random.shuffle(idx_full)

        # 'split' argument is int or float
        # int: number of validation samples
        # float: proportion of validation dataset out of total dataset
        if isinstance(split, int):
            assert split > 0, "Split must be an integer larger than zero."
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # validation&train indices
        valid_idx = idx_full[:len_valid]
        train_idx = np.delete(idx_full, np.arange(0,len_valid))

        # train&validation samplers
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler


    ### Split validation dataset
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

