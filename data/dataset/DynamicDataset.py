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
        self.meta_data = {
            "length_queue": queue.Queue(),
            "idx_queue": queue.Queue(),
            "current_set_idx": 1,
            "current_iteration_idx": 0,
            "total_data_length": 0
        }
        self.data_cache = {}
        self.labels_cache = {}
        self.lock = mp.Lock()
        self.data_queue = queue.Queue()

        self.load_initital_data_into_cache()
        self.load_next_data_thread()
        self.update_len()

    def load_initital_data_into_cache(self):
        for idx in range(1, self.cache_size + 1):
            self.meta_data["length_queue"].put(0)
            self.meta_data["idx_queue"].put(idx)
            self._load_data(idx)

    def _load_data(self, idx):
        data_file = os.path.join(self.cache_data_dir, f'{self.data_idx[idx]}_data.h5')
        with h5py.File(data_file, 'r') as hf:
            self.data_cache[idx] = hf['data'][:]
            self.labels_cache[idx] = hf['labels'][:]

        length = len(self.labels_cache[idx])
        with self.lock:
            lengths = list(self.meta_data["length_queue"].queue)
            lengths[idx - 1] = length
            self.meta_data["length_queue"] = queue.Queue()
            for l in lengths:
                self.meta_data["length_queue"].put(l)

    def load_next_data_thread(self):
        def load_next():
            for idx in range(self.cache_size, len(self.data_idx)):
                with self.lock:
                    if len(self.data_cache) >= self.cache_size:
                        oldest_idx = self.meta_data["idx_queue"].get()
                        self.unload_old_data(oldest_idx)

                    self._load_data(idx)
                    self.meta_data["idx_queue"].put(idx)

        thread = Thread(target=load_next)
        thread.start()

    def unload_old_data(self, idx):
        if idx in self.data_cache:
            del self.data_cache[idx]
            del self.labels_cache[idx]

    def update_len(self):
        with self.lock:
            self.meta_data["total_data_length"] = sum(list(self.meta_data["length_queue"].queue))

    def __len__(self):
        return self.meta_data["total_data_length"]

    def __getitem__(self, index):
        cumulative_length = 0
        current_set_idx = None
        set_index = 0

        # Determine which set the index falls into
        lengths = list(self.meta_data["length_queue"].queue)
        for idx, set_length in enumerate(lengths, 1):
            if index < cumulative_length + set_length:
                current_set_idx = idx
                set_index = index - cumulative_length
                break
            cumulative_length += set_length

        if current_set_idx is None:
            raise IndexError("Index out of range")

        # Fetch data and label
        data = self.data_cache[current_set_idx][set_index]
        label = self.labels_cache[current_set_idx][set_index]

        # Apply any necessary transformations
        label = self.target_transform(label)

        return torch.tensor(data), torch.tensor(label)
