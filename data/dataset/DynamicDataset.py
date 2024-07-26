import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import os
import h5py
import numpy as np
from threading import Thread
import queue
from data.dataset.hz_10000 import DatasetHz10000

class DynamicDatasetHz10000(DatasetHz10000):
    def __init__(self, split: str, config_params: dict, cache_size=5):
        super().__init__(split, config_params)
        self.cache_size = cache_size
        self.data_cache = {}
        self.labels_cache = {}
        self.lock = mp.Lock()
        self.data_queue = queue.Queue()
        self.load_data_into_cache()
        self.load_next_data_thread()

    def load_data_into_cache(self):
        for i in range(min(self.cache_size, len(self.data_idx))):
            self._load_data(i)

    def _load_data(self, idx):
        data_file = os.path.join(self.cache_data_dir, f'{self.data_idx[idx]}_data.h5')
        with h5py.File(data_file, 'r') as hf:
            self.data_cache[idx] = hf['data'][:]
            self.labels_cache[idx] = hf['labels'][:]

    def load_next_data_thread(self):
        def load_next():
            for idx in range(self.cache_size, len(self.data_idx)):
                self.data_queue.put(idx)

        thread = Thread(target=load_next)
        thread.start()

    def unload_old_data(self, idx):
        if idx in self.data_cache:
            del self.data_cache[idx]
            del self.labels_cache[idx]

    def __len__(self):
        return sum(len(data) for data in self.data_cache.values())

    def __getitem__(self, index):
        # Calculate which cache and relative index to fetch
        cache_idx = index // len(next(iter(self.data_cache.values())))
        relative_idx = index % len(next(iter(self.data_cache.values())))

        if cache_idx not in self.data_cache:
            with self.lock:
                if not self.data_queue.empty():
                    next_data_idx = self.data_queue.get()
                    self._load_data(next_data_idx)
                    if len(self.data_cache) > self.cache_size:
                        self.unload_old_data(cache_idx - self.cache_size)

        data = self.data_cache[cache_idx][relative_idx]
        label = self.labels_cache[cache_idx][relative_idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
