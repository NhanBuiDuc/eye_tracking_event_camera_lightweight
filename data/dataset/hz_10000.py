"""
This is a file script used for loading the dataset
"""

import pathlib
from typing import List, Callable, Tuple, Optional, Union
import os
import torch
import pdb
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from utils.ini_30.ini_30_aeadat_processor import read_csv, AedatProcessorLinear
import tonic
from tonic.io import make_structured_array
import json
from tonic.transforms import Compose, ToFrame, MergePolarities, EventDrop, RandomFlipPolarity, Decimation, Denoise, CenterCrop, Downsample
from utils.ini_30.transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP, Downscale
from utils.ini_30.util import get_transforms, get_indexes
import glob
from abc import ABC, abstractmethod
from collections import namedtuple
import struct
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from multiprocessing import Queue, Process
from tqdm import tqdm
events_struct = np.dtype([
    ('t', np.uint32),
    ('x', np.uint16),
    ('y', np.uint16),
    ('p', np.uint8)
])

'Types of data'
Event = namedtuple('Event', 'polarity row col timestamp')
Frame = namedtuple('Frame', 'row col img timestamp')

'Color scheme for event polarity'
color = ['r', 'g']

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path,'**', ext), recursive=True))
    return imgs

def extract_event_components(event_list):
    # Since event_list is reversed, we can directly unpack it into separate components
    pols, xs, ys, ts = event_list[::4], event_list[1::4], event_list[2::4], event_list[3::4]
    return pols, xs, ys, ts

'Reads an event file'
def read_aerdat(filepath):
    with open(filepath, mode='rb') as file:
        file_content = file.read()

    ''' Packet format'''
    packet_format = 'BHHI'                              # pol = uchar, (x,y) = ushort, t = uint32
    packet_size = struct.calcsize('='+packet_format)    # 16 + 16 + 8 + 32 bits => 2 + 2 + 1 + 4 bytes => 9 bytes
    num_events = len(file_content)//packet_size
    extra_bits = len(file_content)%packet_size

    '''Remove Extra Bits'''
    if extra_bits:
        file_content = file_content[0:-extra_bits]

    ''' Unpacking'''
    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    event_list.reverse()

    return event_list

'Parses the filename of the frames'
def get_path_info(path):
    path = path.split('/')[-1]
    filename = os.path.basename(path)
    filename = filename.rsplit('.', 1)[0]
    path_parts = filename.split('_')
    index = int(path_parts[0])
    row = int(path_parts[1])
    col = int(path_parts[2])
    stimulus_type = path_parts[3]
    timestamp = int(path_parts[4])
    return {'index': index, 'row': row, 'col': col, 'stimulus_type': stimulus_type,
            'timestamp': timestamp}

def find_closest_index(df, target, start_time, end_time = None, return_last = False):
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Extract x and y from the target list
    target_x, target_y = target
    
    # Ensure columns are numeric if necessary
    df['row'] = pd.to_numeric(df['row'], errors='coerce')
    df['col'] = pd.to_numeric(df['col'], errors='coerce')
    
    # Filter to include only rows with a timestamp larger than the given timestep
    df_filtered = df[df['timestamp'] > start_time]
    if end_time != None:
        df_filtered = df[df['timestamp'] < end_time]        
    # Ensure that we have rows after timestamp filtering
    if df_filtered.empty:
        return None
    
    # Find rows where either the row or col column differs from target_x or target_y
    df_filtered = df_filtered[
        (df_filtered['row'] != target_x) | (df_filtered['col'] != target_y)
    ]
    
    # If no rows meet the criteria, return None
    if df_filtered.empty:
        return None
    if return_last == True:
        # Return the index of the first row that meets the criteria
        return df_filtered.index[-1]
    return df_filtered.index[0]
    
class GetItemStrategy(ABC):
    @abstractmethod
    def get_item(self, dataset, index):
        pass

class TonicTransformGetItemStrategy(GetItemStrategy):
    def get_item(self, events, labels, dataset, transforms):

        tmp_struct = events
        for transform in transforms:
            if transform == "time_jitter":
                tj_fn = tonic.transforms.TimeJitter(std=100, clip_negative=True)
                tmp_struct = tj_fn(tmp_struct)

            if transform == "uniform_noise":
                un_fn = tonic.transforms.UniformNoise(
                    sensor_size=(dataset.img_width, dataset.img_height, dataset.input_channel),
                    n=1000,
                )
                tmp_struct = un_fn(tmp_struct)

            if transform == "event_drop":
                tj_fn = tonic.transforms.DropEvent(p = 1 / 100)
                tmp_struct = tj_fn(tmp_struct)
        
        events = {
            "xy": np.hstack(
                [tmp_struct["x"].reshape(-1, 1), tmp_struct["y"].reshape(-1, 1)]
            ),
            "p": tmp_struct["p"] * 1,
            "t": tmp_struct["t"],
        }

        event_tensor = events.float()
        labels_tensor = labels.float()
        return event_tensor, labels_tensor, dataset.fixed_window_dt

class StaticWindowGetItemStrategy(GetItemStrategy):
    def get_item(self, events, labels, dataset, transforms):
        
        tmp_struct = events
        for transform in transforms:
            if transform == "time_jitter":
                tj_fn = tonic.transforms.TimeJitter(std=100, clip_negative=True)
                tmp_struct = tj_fn(tmp_struct)

            if transform == "uniform_noise":
                un_fn = tonic.transforms.UniformNoise(
                    sensor_size=(dataset.img_width, dataset.img_height, dataset.input_channel),
                    n=1000,
                )
                tmp_struct = un_fn(tmp_struct)

            if transform == "event_drop":
                tj_fn = tonic.transforms.DropEvent(p=1 / 100)
                tmp_struct = tj_fn(tmp_struct)

        events = {
            "xy": np.hstack(
                [tmp_struct["x"].reshape(-1, 1), tmp_struct["y"].reshape(-1, 1)]
            ),
            "p": tmp_struct["pol"] * 1,
            "t": tmp_struct["t"],
        }
        events, labels = dataset.load_static_window(events, labels)
        # event_tensor = events.float()
        # labels = np.concatenate(labels, axis=0)
        return events, labels, dataset.fixed_window_dt

class DynamicWindowGetItemStrategy(GetItemStrategy):
    def get_item(self, labels, events, dataset, transforms):
        
        tmp_struct = make_structured_array(
            events["xy"][:, 0], events["xy"][:, 1], events["t"], events["p"]
        )
        for transform in transforms:
            if transform == "time_jitter":
                tj_fn = tonic.transforms.TimeJitter(std=100, clip_negative=True)
                tmp_struct = tj_fn(tmp_struct)

            if transform == "uniform_noise":
                un_fn = tonic.transforms.UniformNoise(
                    sensor_size=(self.img_width, self.img_height, self.input_channel),
                    n=1000,
                )
                tmp_struct = un_fn(tmp_struct)

            if transform == "event_drop":
                tj_fn = tonic.transforms.DropEvent(p=1 / 100)
                tmp_struct = tj_fn(tmp_struct)

        events = {
            "xy": np.hstack(
                [tmp_struct["x"].reshape(-1, 1), tmp_struct["y"].reshape(-1, 1)]
            ),
            "p": tmp_struct["p"] * 1,
            "t": tmp_struct["t"],
        }
        events, labels, avg_dt = dataset.load_dynamic_window(events, labels)
        # event_tensor = events.float()
        # labels_tensor = labels.float()
        return events, labels, avg_dt

class DatasetHz10000:
    def __init__(
        self,
        split: str,
        config_params: dict,
    ):
        self.frame_stack = []
        self.event_stack = []
        self.split = split
            
        # self.shuffle = shuffle
        for key, value in config_params.items():
            setattr(self, key, value)
            for sub_key, sub_value in value.items():
                setattr(self, sub_key, sub_value)

        if self.split == "train":
            train_val_idx = list(range(self.train_set_range[0], self.train_set_range[1]))
            #random.shuffle(train_val_idxs)
            for idx in self.val_set_idx:
                if idx in train_val_idx:
                    train_val_idx.remove(idx)
            self.data_idx = train_val_idx
        else:
            self.data_idx = self.val_set_idx

        if self.get_item_strategy == "static_window":
            self.get_item_strategy = StaticWindowGetItemStrategy()
        elif self.get_item_strategy == "dynamic_window":
            self.get_item_strategy = DynamicWindowGetItemStrategy()
        # Create the directory if it doesn't exist
        os.makedirs(self.cache_data_dir, exist_ok=True)

        self.all_data = {}
        self.length_index = {}
        # Initialize the merged arrays
        self.merged_data = []
        self.merged_labels = []
        self.avg_dt = 0
        
    def find_index_list(self, label):

        indexes = []
        # label start and last
        tab_start, tab_last = label.iloc[0], label.iloc[-1]
        # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))
        idx = find_closest_index(label, [0, 0], tab_start["timestamp"], return_last=False)
        start_time = label["timestamp"].iloc[idx]   
        end_time = start_time + self.fixed_window_dt * self.num_bins
        x_label_at_index = label["row"].iloc[idx]  
        y_label_at_index = label["col"].iloc[idx]  
        # append multiple data slices
        indexes.append(idx)

        while(end_time < tab_last["timestamp"]):
            idx = find_closest_index(label, [x_label_at_index, y_label_at_index], end_time, return_last=False) - 2
            try:
                start_time = label["timestamp"].iloc[idx]
                end_time = start_time + self.fixed_window_dt * self.num_bins
                if(end_time < tab_last["timestamp"]):
                    indexes.append(idx)
            except:
                pass
            try:
                start_time = label["timestamp"].iloc[idx+1]
                end_time = start_time + self.fixed_window_dt * self.num_bins
                if(end_time < tab_last["timestamp"]):
                    indexes.append(idx+1)
            except:
                pass
            # start_time = label["timestamp"].iloc[idx+2]
            # end_time = start_time + self.fixed_window_dt * self.num_bins
            # if(end_time < tab_last["timestamp"]):
            #     indexes.append(idx+2)
            try:
                start_time = label["timestamp"].iloc[idx+2]
                end_time = start_time + self.fixed_window_dt * self.num_bins
                if(end_time < tab_last["timestamp"]):
                    indexes.append(idx+2)
            except:
                pass
        return indexes
            
    def prepare_unstructured_data(self):
        length_index = 0
        for idx in self.data_idx:
            print(f"Preparing index for user {idx}")
            # Initialize dictionaries for each idx
            self.all_data[idx] = {}        
            # self.transform, self.target_transform = get_transforms(self.dataset_params, self.training_params)
            left_frame_stack, left_event_stack, left_labels = self.collect_data(idx, 0)
            right_frame_stack, right_event_stack, right_labels = self.collect_data(idx, 1)
            
            left_pols, left_xs, left_ys, left_ts = extract_event_components(left_event_stack)
            right_pols, right_xs, right_ys, right_ts = extract_event_components(right_event_stack)
            max_xs, min_xs = max(left_xs), min(left_xs)
            max_ys, min_ys = max(left_ys), min(left_ys)
            max_ts, min_ts = max(left_ts), min(left_ts)
            print(f"User: {idx}")
            print(f'Max x: {max_xs}, Min x: {min_xs}')
            print(f'Max y: {max_ys}, Min y: {min_ys}')
            print(f'Max t: {max_ts}, Min t: {min_ts}')
            print(f'Max row label: {left_labels["row"].max()}, Max column label: {left_labels["col"].max()}')

            left_eye_data = make_structured_array(left_ts, left_xs, left_ys, left_pols, dtype=events_struct)
            right_eye_data = make_structured_array(right_ts, right_xs, right_ys, right_pols, dtype=events_struct)
            # normalize
            left_eye_data = self.input_transform(left_eye_data)
            right_eye_data = self.input_transform(right_eye_data)

            left_eye_indexes = self.find_index_list(left_labels)
            right_eye_indexes = self.find_index_list(right_labels)
            num_sample = self.num_sample_each_user
            self.all_data[idx] = {
                "left_eye_normalized_data": left_eye_data,
                "right_eye_normalized_data": right_eye_data,
                "left_indexes": left_eye_indexes[:num_sample],
                "right_eye_indexes": right_eye_indexes[:num_sample],
                "left_eye_raw_label_dataframe": left_labels,
                "right_eye_raw_label_dataframe": right_labels,
                "length": length_index + len(left_eye_indexes[:num_sample]) + len(right_eye_indexes[:num_sample])
            }
            length_index = length_index + len(left_eye_indexes[:num_sample]) + len(right_eye_indexes[:num_sample])
            self.length_index[idx] = length_index

    def load_data(self):
        data_path = f"{self.cache_data_dir}/data/{self.split}.npy"
        label_path = f"{self.cache_data_dir}/label/{self.split}.npy"

        self.merged_data = np.load(data_path)
        self.merged_labels = np.load(label_path)

    def __repr__(self):
        return self.__class__.__name__

    def __len__(self):
        return sum(user_data['length'] for user_data in self.all_data.values())

    def __getitem__(self, index):
        user_idx = None
        previous_length = 0

        for idx in self.data_idx:
            if index < self.length_index[idx]:
                user_idx = idx
                break
            previous_length = self.length_index[idx]

        if user_idx is None:
            raise IndexError("Index out of range")

        relative_index = index - previous_length
        print("Train User ", user_idx)
        left_indexes = self.all_data[user_idx]["left_indexes"]
        right_indexes = self.all_data[user_idx]["right_eye_indexes"]

        if relative_index < len(left_indexes):
            # data = left_eye_data[left_indexes[relative_index]]
            eye_data = self.all_data[user_idx]["left_eye_normalized_data"]
            raw_label_dataframe = self.all_data[user_idx]["left_eye_raw_label_dataframe"]
            timestamp_index = left_indexes[relative_index]          
        else:
            relative_index -= len(left_indexes)
            eye_data = self.all_data[user_idx]["right_eye_normalized_data"]
            raw_label_dataframe = self.all_data[user_idx]["right_eye_raw_label_dataframe"]
            timestamp_index = right_indexes[relative_index]     

        data, label = self.get_item(eye_data, raw_label_dataframe, timestamp_index)
        label = self.target_transform(label)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return data, label
        
    def process_data_for_idx(self, idx, result_queue):
        # Initialize dictionaries for each idx
        self.all_data[idx] = {}
        self.all_labels[idx] = {}            
        # self.transform, self.target_transform = get_transforms(self.dataset_params, self.training_params)
        left_frame_stack, left_event_stack, left_labels = self.collect_data(idx, 0)
        right_frame_stack, right_event_stack, right_labels = self.collect_data(idx, 1)
        
        left_pols, left_xs, left_ys, left_ts = extract_event_components(left_event_stack)
        right_pols, right_xs, right_ys, right_ts = extract_event_components(right_event_stack)
        max_xs, min_xs = max(left_xs), min(left_xs)
        max_ys, min_ys = max(left_ys), min(left_ys)
        max_ts, min_ts = max(left_ts), min(left_ts)
        print(f"User: {idx}")
        print(f'Max x: {max_xs}, Min x: {min_xs}')
        print(f'Max y: {max_ys}, Min y: {min_ys}')
        print(f'Max t: {max_ts}, Min t: {min_ts}')
        print(f'Max row label: {left_labels["row"].max()}, Max column label: {left_labels["col"].max()}')

        left_eye_data = make_structured_array(left_ts, left_xs, left_ys, left_pols, dtype=events_struct)
        right_eye_data = make_structured_array(right_ts, right_xs, right_ys, right_pols, dtype=events_struct)
        # normalize
        left_eye_data = self.input_transform(left_eye_data)
        right_eye_data = self.input_transform(right_eye_data)

        left_eye_data, left_labels, fixed_window_dt = self.get_item_strategy.get_item(left_eye_data, left_labels, self, self.tonic_transforms)
        right_eye_data, right_labels, fixed_window_dt = self.get_item_strategy.get_item(right_eye_data, right_labels, self, self.tonic_transforms)

        left_labels = self.target_transform(left_labels)
        right_labels = self.target_transform(right_labels)

        left_train_data, left_train_label, left_val_data, left_val_label, left_test_data, left_test_label = self.split_data(left_eye_data, left_labels, self.split_ratio, 42, "left", idx)
        right_train_data, right_train_label, right_val_data, right_val_label, right_test_data, right_test_label = self.split_data(right_eye_data, right_labels, self.split_ratio, 42, "right", idx)

        # result_queue.put([(left_train_data, left_train_label), (right_train_data, right_train_label)])
        # result_queue.put([(left_val_data, left_val_label), (right_val_data, right_val_label)])
        # self.all_data[idx]["left_data"] = left_eye_data
        # self.all_labels[idx]["left_label"] = left_labels
        # self.all_data[idx]["right_data"] = right_eye_data
        # self.all_labels[idx]["right_label"] = right_labels

        result_queue.put((left_train_data, left_train_label, left_val_data, left_val_label, left_test_data, left_test_label, 
                            right_train_data, right_train_label, right_val_data, right_val_label, right_test_data, right_test_label))

    def merge_results(self, results):
        merged_train_data = []
        merged_train_labels = []
        merged_val_data = []
        merged_val_labels = []
        merged_test_data = []
        merged_test_labels = []
        
        for result in results:
            (left_train_data, left_train_label, left_val_data, left_val_label, left_test_data, left_test_label, 
            right_train_data, right_train_label, right_val_data, right_val_label, right_test_data, right_test_label) = result
            
            merged_train_data.extend([left_train_data, right_train_data])
            merged_train_labels.extend([left_train_label, right_train_label])
            merged_val_data.extend([left_val_data, right_val_data])
            merged_val_labels.extend([left_val_label, right_val_label])
            merged_test_data.extend([left_test_data, right_test_data])
            merged_test_labels.extend([left_test_label, right_test_label])
        
        return (np.array(merged_train_data), np.array(merged_train_labels), 
                np.array(merged_val_data), np.array(merged_val_labels), 
                np.array(merged_test_data), np.array(merged_test_labels))

    def parallel_process_data(self):
        # Create a queue to collect results
        result_queue = Queue()

        # Create and start a process for each idx
        processes = []
        for idx in self.data_idx:
            process = Process(target=partial(self.process_data_for_idx, idx, result_queue))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Collect results from the queue
        results = []
        while not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        # Merge results
        merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, merged_test_data, merged_test_labels = self.merge_results(results)

        os.makedirs(f"{self.cache_data_dir}/data", exist_ok=True)
        os.makedirs(f"{self.cache_data_dir}/label", exist_ok=True)

        # Save all the merged data splits as NumPy arrays
        np.save(f"{self.cache_data_dir}/data/train.npy", merged_train_data)
        np.save(f"{self.cache_data_dir}/label/train.npy", merged_train_labels)

        np.save(f"{self.cache_data_dir}/data/val.npy", merged_val_data)
        np.save(f"{self.cache_data_dir}/label/val.npy", merged_val_labels)

        np.save(f"{self.cache_data_dir}/data/test.npy", merged_test_data)
        np.save(f"{self.cache_data_dir}/label/test.npy", merged_test_labels)

        # Optionally, return them if needed
        return merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, merged_test_data, merged_test_labels

    def split_data(self, data, label, ratio, seed, eye, user_idx):
        train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=1-ratio, random_state=seed)
        val_data, test_data, val_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5, random_state=seed)
        
        
        # os.makedirs(f"{self.cache_data_dir}/train/data", exist_ok=True)
        # os.makedirs(f"{self.cache_data_dir}/train/label", exist_ok=True)

        # os.makedirs(f"{self.cache_data_dir}/val/data", exist_ok=True)
        # os.makedirs(f"{self.cache_data_dir}/val/label", exist_ok=True)

        # os.makedirs(f"{self.cache_data_dir}/test/data", exist_ok=True)
        # os.makedirs(f"{self.cache_data_dir}/test/label", exist_ok=True)

        # np.save(f"{self.cache_data_dir}/train/data/user{user_idx}_{eye}.npy", train_data)
        # np.save(f"{self.cache_data_dir}/train/label/user{user_idx}_{eye}.npy", train_label)

        # np.save(f"{self.cache_data_dir}/val/data/user{user_idx}_{eye}.npy", val_data)
        # np.save(f"{self.cache_data_dir}/val/label/user{user_idx}_{eye}.npy", val_label)

        # np.save(f"{self.cache_data_dir}/test/data/user{user_idx}_{eye}.npy", test_data)
        # np.save(f"{self.cache_data_dir}/test/label/user{user_idx}_{eye}.npy", test_label)

        return train_data, train_label, val_data, val_label, test_data, test_label
    
    def cache_files_exist(self, user_idx, eye):
        """Check if all cache files for a given user and eye exist."""
        required_files = [
            f"{self.cache_data_dir}/train/data/user{user_idx}_{eye}.npy",
            f"{self.cache_data_dir}/train/label/user{user_idx}_{eye}.npy",
            f"{self.cache_data_dir}/val/data/user{user_idx}_{eye}.npy",
            f"{self.cache_data_dir}/val/label/user{user_idx}_{eye}.npy",
            f"{self.cache_data_dir}/test/data/user{user_idx}_{eye}.npy",
            f"{self.cache_data_dir}/test/label/user{user_idx}_{eye}.npy"
        ]
        return all(os.path.exists(file) for file in required_files)

    def find_first_n_unique_pairs(self, events, N):
        seen_pairs = set()
        result = []
        seen = 0

        for i in reversed(range(len(events))):
            event_tuple = tuple(events[i])

            if event_tuple not in seen_pairs:
                seen_pairs.add(event_tuple)
                result.append(events[i])
                seen += 1
                if seen == N:
                    break
            else:
                result.append(events[i])

        return np.array(result)
    
    def get_item(self, data, label, idx):
        data = {
            "xy": np.hstack(
                [data["x"].reshape(-1, 1), data["y"].reshape(-1, 1)]
            ),
            "p": data["p"] * 1,
            "t": data["t"],
        }
        start_time = label["timestamp"].iloc[idx]   
        end_time = start_time + self.fixed_window_dt * self.num_bins

        # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))
        evs_t = data["t"][data["t"] >= start_time]
        evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

        # frame
        data_temp = np.zeros(
            (self.num_bins, self.input_channel, self.img_width, self.img_height)
        )

        # indexes
        start_idx = 0

        # get intermediary labels based on num of bins
        fixed_timestamps = np.linspace(start_time, end_time, self.num_bins)
        x_axis, y_axis = [], []
        for i, fixed_tmp in enumerate(fixed_timestamps):

            # label
            idx = np.searchsorted(label["timestamp"], fixed_tmp, side="left")

            # Weighted interpolation
            try:
                t0 = label["timestamp"].iloc[idx - 1]
                t1 = label["timestamp"].iloc[idx]

                weight0 = (t1 - fixed_tmp) / (t1 - t0)
                weight1 = (fixed_tmp - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        label.iloc[idx - 1]["row"] * weight0
                        + label.iloc[idx]["row"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        label.iloc[idx - 1]["col"] * weight0
                        + label.iloc[idx]["col"] * weight1
                    )
                )
            except:
                x_axis.append(
                int(
                    label.iloc[idx - 1]["row"]
                )
                )
                y_axis.append(
                    int(
                        label.iloc[idx - 1]["col"] * weight0
                    )
                )
            # slice
            t = evs_t[start_idx:][evs_t[start_idx:] <= fixed_tmp]
            # if t.shape[0] == 0:
            #     continue
            xy = evs_xy[start_idx : start_idx + t.shape[0], :]
            p = evs_p[start_idx : start_idx + t.shape[0]]

            np.add.at(data_temp[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
            if self.input_channel > 1:
                np.add.at(
                    data_temp[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                )
                data_temp[i, 0, :, :][
                    data_temp[i, 1, :, :] >= data_temp[i, 0, :, :]
                ] = 0  # if ch 1 has more evs than 0
                data_temp[i, 1, :, :][
                    data_temp[i, 1, :, :] < data_temp[i, 0, :, :]
                ] = 0  # if ch 0 has more evs than 1

            data_temp[i] = data_temp[i].clip(0, 1)  # no double events

            # move pointers
            start_idx += t.shape[0]

        return np.array(data_temp).astype(np.float32), np.array(np.column_stack((x_axis, y_axis))).astype(np.float32)
        
    def load_static_window(self, data, labels):
        # label start and last
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))
        idx = find_closest_index(labels, [0, 0], tab_start["timestamp"], return_last=False)
        start_time = labels["timestamp"].iloc[idx]   
        end_time = start_time + self.fixed_window_dt * self.num_bins

        # append multiple data slices
        batch_data = []
        batch_label = []
        switch = 0
        while(end_time < tab_last["timestamp"]):

            # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
            # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))
            evs_t = data["t"][data["t"] >= start_time]
            evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

            # frame
            data_temp = np.zeros(
                (self.num_bins, self.input_channel, self.img_width, self.img_height)
            )

            # indexes
            start_idx = 0

            # get intermediary labels based on num of bins
            fixed_timestamps = np.linspace(start_time, end_time, self.num_bins)
            x_axis, y_axis = [], []
            for i, fixed_tmp in enumerate(fixed_timestamps):

                # label
                idx = np.searchsorted(labels["timestamp"], fixed_tmp, side="left")
                # row = int(labels.iloc[idx]["row"])
                # col = int(labels.iloc[idx]["col"])
                # if row == 0 and col == 0:
                #     skip = True
                #     break

                # Weighted interpolation
                t0 = labels["timestamp"].iloc[idx - 1]
                t1 = labels["timestamp"].iloc[idx]

                weight0 = (t1 - fixed_tmp) / (t1 - t0)
                weight1 = (fixed_tmp - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        labels.iloc[idx - 1]["row"] * weight0
                        + labels.iloc[idx]["row"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["col"] * weight0
                        + labels.iloc[idx]["col"] * weight1
                    )
                )

                # slice
                t = evs_t[start_idx:][evs_t[start_idx:] <= fixed_tmp]
                # if t.shape[0] == 0:
                #     continue
                xy = evs_xy[start_idx : start_idx + t.shape[0], :]
                p = evs_p[start_idx : start_idx + t.shape[0]]

                np.add.at(data_temp[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
                if self.input_channel > 1:
                    np.add.at(
                        data_temp[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                    )
                    data_temp[i, 0, :, :][
                        data_temp[i, 1, :, :] >= data_temp[i, 0, :, :]
                    ] = 0  # if ch 1 has more evs than 0
                    data_temp[i, 1, :, :][
                        data_temp[i, 1, :, :] < data_temp[i, 0, :, :]
                    ] = 0  # if ch 0 has more evs than 1

                data_temp[i] = data_temp[i].clip(0, 1)  # no double events

                # move pointers
                start_idx += t.shape[0]  
            
            batch_data.append(data_temp)
            batch_label.append(np.column_stack((x_axis, y_axis)))
            try:
                # # Equal head and tail

                if switch == 0:
                    idx = find_closest_index(labels, [x_axis[-1], y_axis[-1]], return_last=False) - 2
                    start_time = labels["timestamp"].iloc[idx]   
                    end_time = start_time + self.fixed_window_dt * self.num_bins
                # # small head and big tail

                # elif switch == 1:
                #     idx = idx + 1
                #     start_time = labels["timestamp"].iloc[idx]   
                #     end_time = start_time + self.fixed_window_dt * self.num_bins
                #     switch = 2

                # # Mono data
                # elif switch == 2:
                #     idx = idx + 1
                #     start_time = labels["timestamp"].iloc[idx]   
                #     end_time = start_time + self.fixed_window_dt * self.num_bins
                #     switch = 3
                # # big head and small tail
                # elif switch == 3:
                #     idx = idx + 2
                #     start_time = labels["timestamp"].iloc[idx]   
                #     end_time = start_time + self.fixed_window_dt * self.num_bins
                #     switch = 0

            except:
                start_time = end_time
                end_time = start_time + self.fixed_window_dt * self.num_bins                

            # if skip == False:             

            # else:
            #     # idx = find_closest_index(labels, [0, 0], end_time, return_last=False)
            #     # start_time = labels["timestamp"].iloc[idx]   
            #     # end_time = start_time + self.fixed_window_dt * self.num_bins
            #     #                 idx = np.searchsorted(labels["timestamp"], fixed_tmp, side="left")
            #     pass
        # batch_data = torch.stack(batch_data)
        # batch_label = torch.stack(batch_label)
            # break
        return np.array(batch_data).astype(np.float32), np.array(batch_label).astype(np.float32)

    # def load_static_window(self, data, labels):
        # tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))

        # start_time = tab_last["timestamp"] - self.fixed_window_dt * (self.num_bins + 1)
        # evs_t = data["t"][data["t"] >= start_time]
        # evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

        # # frame
        # data = np.zeros(
        #     (self.num_bins, self.input_channel, self.img_width, self.img_height)
        # )

        # # indexes
        # start_idx = 0

        # # get intermediary labels based on num of bins
        # fixed_timestamps = np.linspace(start_time, tab_last["timestamp"], self.num_bins)
        # x_axis, y_axis = [], []

        # for i, fixed_tmp in enumerate(fixed_timestamps):
        #     # label
        #     idx = np.searchsorted(labels["timestamp"], fixed_tmp, side="left")
        #     if idx == 0:
        #         x_axis.append(start_label[0])
        #         y_axis.append(start_label[1])
        #     elif idx == len(labels["timestamp"]):
        #         x_axis.append(end_label[0])
        #         y_axis.append(end_label[1])
        #     else:  # Weighted interpolation
        #         t0 = labels["timestamp"].iloc[idx - 1]
        #         t1 = labels["timestamp"].iloc[idx]

        #         weight0 = (t1 - fixed_tmp) / (t1 - t0)
        #         weight1 = (fixed_tmp - t0) / (t1 - t0)

        #         x_axis.append(
        #             int(
        #                 labels.iloc[idx - 1]["row"] * weight0
        #                 + labels.iloc[idx]["row"] * weight1
        #             )
        #         )
        #         y_axis.append(
        #             int(
        #                 labels.iloc[idx - 1]["col"] * weight0
        #                 + labels.iloc[idx]["col"] * weight1
        #             )
        #         )

        #     # slice
        #     t = evs_t[start_idx:][evs_t[start_idx:] <= fixed_tmp]
        #     if t.shape[0] == 0:
        #         continue
        #     xy = evs_xy[start_idx : start_idx + t.shape[0], :]
        #     p = evs_p[start_idx : start_idx + t.shape[0]]

        #     np.add.at(data[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
        #     if self.input_channel > 1:
        #         np.add.at(
        #             data[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
        #         )
        #         data[i, 0, :, :][
        #             data[i, 1, :, :] >= data[i, 0, :, :]
        #         ] = 0  # if ch 1 has more evs than 0
        #         data[i, 1, :, :][
        #             data[i, 1, :, :] < data[i, 0, :, :]
        #         ] = 0  # if ch 0 has more evs than 1

        #     data[i] = data[i].clip(0, 1)  # no double events

        #     # move pointers
        #     start_idx += t.shape[0]

        # # frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        # # frames = frames.permute(0, 1, 3, 2) 
        # # labels = self.target_transform(np.vstack([x_axis, y_axis]))

        # # self.avg_dt += (evs_t[-1] - evs_t[0]) / self.num_bins
        # # self.items += 1
        # labels = np.column_stack((x_axis, y_axis))
        # return data, labels

    def load_dynamic_window(self, data, labels):
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        end_label = (int(tab_last.row.item()), int(tab_last.col.item()))

        evs_t, evs_p, evs_xy = data["t"], data["p"], data["xy"]

        # frame
        data = np.zeros(
            (self.num_bins, self.input_channel, self.img_width, self.img_height)
        )

        # label
        x_axis, y_axis = [], []

        # indexes
        start_idx, end_idx = 0, len(evs_p) - 1
        for i in reversed(range(self.num_bins)):
            # find self.events_per_frame events
            xy = self.find_first_n_unique_pairs(
                evs_xy[:end_idx, :], self.events_per_frame
            )
            start_idx = end_idx - len(xy)
            p = evs_p[start_idx: ]

            np.add.at(data[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
            if self.input_channel > 1:
                np.add.at(
                    data[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                )
                # turn off channel 0 events if channel 1 has more events
                data[i, 0, :, :][
                    data[i, 1, :, :] >= data[i, 0, :, :]
                ] = 0  # if ch 1 has more evs than ch 0
                # turn off channel 1 events if channel 1 has more events
                data[i, 1, :, :][
                    data[i, 1, :, :] < data[i, 0, :, :]
                ] = 0  # if ch 0 has more evs than ch 1

            data[i] = data[i].clip(0, 1)

            # label time
            label_time = (evs_t[start_idx] + evs_t[end_idx]) / 2

            # move pointers
            end_idx = start_idx

            # label
            idx = np.searchsorted(labels["timestamp"], label_time, side="left")
            if idx == 0:
                x_axis.append(start_label[0])
                y_axis.append(start_label[1])
            elif idx == len(labels["timestamp"]):
                x_axis.append(end_label[0])
                y_axis.append(end_label[1])
            else:  # Weighted interpolation
                t0 = labels["timestamp"].iloc[idx - 1]
                t1 = labels["timestamp"].iloc[idx]

                weight0 = (t1 - label_time) / (t1 - t0)
                weight1 = (label_time - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        labels.iloc[idx - 1]["row"] * weight0
                        + labels.iloc[idx]["row"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["col"] * weight0
                        + labels.iloc[idx]["col"] * weight1
                    )
                )
        frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        frames = frames.permute(0, 1, 3, 2)
        x_axis.reverse()
        y_axis.reverse()
        x_axis = torch.tensor(x_axis)
        y_axis = torch.tensor(y_axis)
        labels = torch.stack((x_axis, y_axis), dim=1)
        avg_dt = (evs_t[-1] - evs_t[start_idx + len(xy)]) / self.num_bins

        return frames, labels, avg_dt
    'Loads in data from the data_dir as filenames'
    def collect_data(self, user_id, eye=0):
        print('Loading Frames....')
        frame_stack, labels = self.load_frame_data(user_id, eye)
        print('There are ' + str(len(frame_stack)) + ' frames \n')
        print('Loading Events....')
        event_stack = self.load_event_data(user_id, eye)
        print('There are ' + str(len(event_stack)) + ' events \n')
        return frame_stack, event_stack, labels
    
    def load_frame_data(self, user_id, eye):
        filepath_list = []
        user_name = "user" + str(user_id)
        img_dir = os.path.join(self.data_dir, user_name, str(eye), 'frames')
        img_filepaths = list(glob_imgs(img_dir))
        img_filepaths.sort(key=lambda name: get_path_info(name)['index'])
        label_list = []
        # img_filepaths.reverse()
        for fpath in img_filepaths:
            path_info = get_path_info(fpath)
            label_list.append(path_info)
            frame = Frame(path_info['row'], path_info['col'], fpath, path_info['timestamp'])
            filepath_list.append(frame)
        labels = pd.DataFrame(label_list)
        return filepath_list, labels

    def load_event_data(self, user_id, eye):
        user_name = "user" + str(user_id)
        event_file = os.path.join(self.data_dir, user_name, str(eye), 'events.aerdat')
        filepath_list = read_aerdat(event_file)
        filepath_list.reverse()
        return filepath_list
    def input_transform(self, input):
        max_x_range = self.davis_sensor_size[0]
        max_y_range = self.davis_sensor_size[1]
        desired_x = self.img_width
        desired_y = self.img_height
        scale_x = desired_x / max_x_range
        scale_y = desired_y / max_y_range
        input["x"] = (input["x"] * scale_x).astype(int)
        input["y"] = (input["y"] * scale_y).astype(int)
        print("Normalized input maximum value: ", input["x"].max())
        print("Normalized label maximum value: ", input["y"].max())
        return input
    
    def target_transform(self, target):


        """
        Normalize a list of numpy arrays to the range [0, 1] for each dimension separately.
        
        Parameters:
            arrays (list of numpy.ndarray): List of numpy arrays each with shape (64, 2).
        
        Returns:
            list of numpy.ndarray: List of normalized numpy arrays.
        """

        # Ensure that the list is not empty
        # if not target:
        #     raise ValueError("The list of arrays is empty.")
        
        # Normalize each array
        target[:, 0] = target[:, 0] / self.stimulus_screen_size[0]
        target[:, 1] = target[:, 1] / self.stimulus_screen_size[1]

        return target