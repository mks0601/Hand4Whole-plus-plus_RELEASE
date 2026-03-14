import random
import numpy as np
from torch.utils.data.dataset import Dataset

class MultipleDatasets(Dataset):
    def __init__(self, dbs, db_sample_prob):
        self.dbs = dbs
        self.db_sample_prob = db_sample_prob

    def __len__(self):
        return int(sum([len(db) for db in self.dbs]) / len(self.dbs))

    def __getitem__(self, index):
        db_idx = int(np.random.choice(np.arange(len(self.dbs)), size=1, replace=False, p=self.db_sample_prob))
        data_idx = random.randint(0,len(self.dbs[db_idx])-1)
        return self.dbs[db_idx][data_idx]
