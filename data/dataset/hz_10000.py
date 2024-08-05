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
import multiprocessing as mp
import numpy as np
import concurrent.futures
from threading import Lock
import multiprocessing
import h5py

# def process_and_save_user_data(args):
#     idx, self, lock, txt_file = args
#     print(f"Preparing index for user {idx}")
#     self.all_data[idx] = {}

#     left_frame_stack, left_event_stack, left_labels = self.collect_data(idx, 0)
#     right_frame_stack, right_event_stack, right_labels = self.collect_data(idx, 1)

#     left_pols, left_xs, left_ys, left_ts = extract_event_components(left_event_stack)
#     right_pols, right_xs, right_ys, right_ts = extract_event_components(right_event_stack)

#     left_eye_data = make_structured_array(left_ts, left_xs, left_ys, left_pols, dtype=events_struct)
#     right_eye_data = make_structured_array(right_ts, right_xs, right_ys, right_pols, dtype=events_struct)

#     left_eye_data = self.input_transform(left_eye_data)
#     right_eye_data = self.input_transform(right_eye_data)

#     left_eye_indexes = self.find_index_list(left_labels)
#     right_eye_indexes = self.find_index_list(right_labels)

#     for timestamp_index in tqdm(left_eye_indexes):
#         data, label = self.get_item(left_eye_data, left_labels, timestamp_index)
#         data_array = np.array(data)
#         label_array = np.array(label)

#         left_data_filename = f'{self.cache_data_dir}/{idx}_left_{timestamp_index}_data.h5'
#         left_label_filename = f'{self.cache_data_dir}/{idx}_left_{timestamp_index}_label.h5'

#         with h5py.File(left_data_filename, 'w') as hf:
#             hf.create_dataset('data', data=data_array, compression='gzip')

#         with h5py.File(left_label_filename, 'w') as hf:
#             hf.create_dataset('labels', data=label_array, compression='gzip')

#         with lock:
#             with open(txt_file, 'a') as f:
#                 f.write(f'{left_data_filename}\n')

#     for timestamp_index in tqdm(right_eye_indexes):
#         data, label = self.get_item(right_eye_data, right_labels, timestamp_index)
#         data_array = np.array(data)
#         label_array = np.array(label)

#         right_data_filename = f'{self.cache_data_dir}/{idx}_right_{timestamp_index}_data.h5'
#         right_label_filename = f'{self.cache_data_dir}/{idx}_right_{timestamp_index}_label.h5'

#         with h5py.File(right_data_filename, 'w') as hf:
#             hf.create_dataset('data', data=data_array, compression='gzip')

#         with h5py.File(right_label_filename, 'w') as hf:
#             hf.create_dataset('labels', data=label_array, compression='gzip')

#         with lock:
#             with open(txt_file, 'a') as f:
#                 f.write(f'{right_data_filename}\n')

#     return idx


def process_and_save_user_data(args):
    idx, self, lock, txt_file = args
    print(f"Preparing index for user {idx}")
    self.all_data[idx] = {}

    left_frame_stack, left_event_stack, left_labels = self.collect_data(idx, 0)
    right_frame_stack, right_event_stack, right_labels = self.collect_data(idx, 1)

    left_pols, left_xs, left_ys, left_ts = extract_event_components(left_event_stack)
    right_pols, right_xs, right_ys, right_ts = extract_event_components(right_event_stack)

    left_eye_data = make_structured_array(left_ts, left_xs, left_ys, left_pols, dtype=events_struct)
    right_eye_data = make_structured_array(right_ts, right_xs, right_ys, right_pols, dtype=events_struct)

    left_eye_data = self.input_transform(left_eye_data)
    right_eye_data = self.input_transform(right_eye_data)

    self.window_size_event_label_sync(left_eye_data, left_labels, idx, "left", txt_file, lock)
    self.window_size_event_label_sync(right_eye_data, right_labels, idx, "right", txt_file, lock)
     
    return idx

def find_closest_index(df, target, start_time=None, end_time=None, return_last=False):
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Extract target row and col
    target_x, target_y = target
    
    # Ensure columns are numeric if necessary
    df['row'] = pd.to_numeric(df['row'], errors='coerce')
    df['col'] = pd.to_numeric(df['col'], errors='coerce')
    
    # Filter the DataFrame based on the given timestamps
    if start_time is not None and end_time is not None:
        df_filtered = df[(df['timestamp'] > start_time) & (df['timestamp'] < end_time)]
    elif start_time is not None:
        df_filtered = df[df['timestamp'] > start_time]
    elif end_time is not None:
        df_filtered = df[df['timestamp'] < end_time]
    else:
        df_filtered = df
    
    # Find rows where either the row or col column differs from target_x or target_y
    df_filtered = df_filtered[
        (df_filtered['row'] != target_x) | (df_filtered['col'] != target_y)
    ]
    
    # If no rows meet the criteria, return None
    if df_filtered.empty:
        return None

    # Return the index of the first or last row that meets the criteria
    if return_last:
        return df_filtered.index[-1]
    return df_filtered.index[0]

def find_index_with_different_stimulus(df, start_index, stimulus):
    # Ensure start_index is within the range of the DataFrame
    if start_index < 0 or start_index >= len(df):
        raise IndexError("start_index is out of the DataFrame's range.")
    
    # Iterate from the start_index to the end of the DataFrame
    for i in range(start_index + 1, len(df)):
        if df.loc[i, 'stimulus_type'] != stimulus:
            return i

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

        # Create the directory if it doesn't exist
        os.makedirs(self.cache_data_dir, exist_ok=True)

        self.all_data = {}
        self.length_index = {}
        # Initialize the merged arrays
        self.merged_data = []
        self.merged_labels = []
        self.avg_dt = 0
    
    # def prepare_unstructured_data(self, data_idx=None):
    #     if data_idx is None:
    #         data_idx = self.data_idx

    #     manager = mp.Manager()
    #     lock = manager.Lock()

    #     txt_file = f'{self.cache_data_dir}/{self.split}.txt'

    #     # Remove the text file if it already exists
    #     if os.path.exists(txt_file):
    #         os.remove(txt_file)

    #     pool = mp.Pool(mp.cpu_count())
    #     args = [(idx, self, lock, txt_file) for idx in data_idx]

    #     pool.map(process_and_save_user_data, args)

    #     pool.close()
    #     pool.join()


    def prepare_unstructured_data(self, reset = False, data_idx=None):
        if data_idx is None:
            data_idx = self.data_idx

        for idx in data_idx:
            folder = f"{self.annotation_dir}/user_{idx}"
            # Create the folder if it does not exist
            os.makedirs(folder, exist_ok=True)

            print(f"Folder '{folder}' is ready.")

            left_txt_file = os.path.join(folder, f"{self.split}_left.txt")
            right_txt_file = os.path.join(folder, f"{self.split}_right.txt")

            if reset:
                if os.path.exists(left_txt_file):
                    os.remove(left_txt_file)
                if os.path.exists(right_txt_file):
                    os.remove(right_txt_file)

            print(f"Preparing index for user {idx}")
            self.all_data[idx] = {}

            left_frame_stack, left_event_stack, left_labels = self.collect_data(idx, 0)
            right_frame_stack, right_event_stack, right_labels = self.collect_data(idx, 1)

            left_pols, left_xs, left_ys, left_ts = extract_event_components(left_event_stack)
            right_pols, right_xs, right_ys, right_ts = extract_event_components(right_event_stack)

            left_eye_data = make_structured_array(left_ts, left_xs, left_ys, left_pols, dtype=events_struct)
            right_eye_data = make_structured_array(right_ts, right_xs, right_ys, right_pols, dtype=events_struct)

            left_eye_data = self.input_transform(left_eye_data)
            right_eye_data = self.input_transform(right_eye_data)

            self.window_size_event_label_sync(left_eye_data, left_labels, idx, "left", left_txt_file)
            self.window_size_event_label_sync(right_eye_data, right_labels, idx, "right", right_txt_file)


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


    def _load_file(self, data_file, label_file):
        print("Load data", data_file)
        print("Load label", label_file)
        
        with h5py.File(data_file, 'r') as hf:
            user_data = hf['data'][:]
            print("Finished data loading ", data_file)
        
        with h5py.File(label_file, 'r') as hf:
            user_labels = hf['labels'][:]
            print("Finished label loading ", label_file)
            
        return user_data, user_labels
        
    def load_cached_data(self, data_idx=None):
        if data_idx is not None:
            self.data_idx = data_idx

        # Split the data indices into two halves
        mid_point = len(self.data_idx) // 2
        first_half_idx = self.data_idx[:mid_point]
        second_half_idx = self.data_idx[mid_point:]

        # Helper function to load and concatenate data and labels for given indices
        def load_and_concatenate(indices):
            data_files = [os.path.join(self.cache_data_dir, f"{index}_data.h5") for index in indices]
            label_files = [os.path.join(self.cache_data_dir, f"{index}_data.h5") for index in indices]

            with multiprocessing.Pool() as pool:
                results = pool.starmap(self._load_file, zip(data_files, label_files))

            all_data = [res[0] for res in results]
            all_labels = [res[1] for res in results]

            return np.concatenate(all_data), np.concatenate(all_labels)

        # Load data and labels for the first half
        self.all_data_first_half, self.all_labels_first_half = load_and_concatenate(first_half_idx)

        # Load data and labels for the second half
        self.all_data_second_half, self.all_labels_second_half = load_and_concatenate(second_half_idx)

    # def _load_file(self, data_file, label_file):
    #     print("Load ", data_file)
    #     print("Load ", label_file)
    #     user_data = np.load(data_file)
    #     print("Finished loading ", data_file)  
    #     user_labels = np.load(label_file)
    #     print("Finished loading ", label_file)        
    #     return user_data, user_labels

    # def load_cached_data(self, data_idx=None):

    #     if data_idx is not None:
    #         self.data_idx = data_idx

    #     # Split the data indices into two halves
    #     mid_point = len(self.data_idx) // 2
    #     first_half_idx = self.data_idx[:mid_point]
    #     second_half_idx = self.data_idx[mid_point:]

    #     # Helper function to load and concatenate data and labels for given indices
    #     def load_and_concatenate(indices):
    #         data_files = [os.path.join(self.cache_data_dir, f"{index}_data.npy") for index in indices]
    #         label_files = [os.path.join(self.cache_data_dir, f"{index}_labels.npy") for index in indices]

    #         with multiprocessing.Pool() as pool:
    #             results = pool.starmap(self._load_file, zip(data_files, label_files))

    #         all_data = [res[0] for res in results]
    #         all_labels = [res[1] for res in results]

    #         return np.concatenate(all_data), np.concatenate(all_labels)

    #     # Load data and labels for the first half
    #     all_data_first_half, all_labels_first_half = load_and_concatenate(first_half_idx)

    #     # Load data and labels for the second half
    #     all_data_second_half, all_labels_second_half = load_and_concatenate(second_half_idx)

    #     # Concatenate the results from both halves
    #     self.all_event = np.concatenate([all_data_first_half, all_data_second_half])
    #     self.all_label = np.concatenate([all_labels_first_half, all_labels_second_half])

    def __repr__(self):
        return self.__class__.__name__
    
    def read_file_list(self):
        self.file_list = []
        for idx in self.data_idx:
            txt_file = f'{self.annotation_dir}/user_{idx}/{self.split}_left.txt'
            with open(txt_file, 'r') as f:
                self.file_list.extend(f.read().splitlines())
            txt_file = f'{self.annotation_dir}/user_{idx}/{self.split}_right.txt'
            with open(txt_file, 'r') as f:
                self.file_list.extend(f.read().splitlines())      
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data_filename = self.file_list[index]
        label_filename = data_filename.replace('_data.h5', '_label.h5')

        with h5py.File(data_filename, 'r') as hf:
            event = hf['data'][:]

        with h5py.File(label_filename, 'r') as hf:
            label = hf['labels'][:]

        if self.target_transform is not None:
            label = label.astype(np.float32)
            label = np.abs(label)  # Convert all values to positive
            label = self.target_transform(label)

        data = torch.tensor(event)
        label = torch.tensor(label)

        return data, label

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

    def window_size_event_label_sync(self, data, labels, user_id, eye, txt_file):
        data = {
            "xy": np.hstack(
                [data["x"].reshape(-1, 1), data["y"].reshape(-1, 1)]
            ),
            "p": data["p"] * 1,
            "t": data["t"],
        }
        # label start and last
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        # start_label = (int(tab_start.row.item()), int(tab_start.col.item()))
        # end_label = (int(tab_last.row.item()), int(tab_last.col.item()))
        idx = find_closest_index(labels, [0, 0], return_last=False)
        start_time = labels["timestamp"].iloc[idx]   
        old_row = labels.iloc[idx]["row"]
        old_col = labels.iloc[idx]["col"]
        old_stimulus = labels.iloc[idx]["stimulus_type"]

        end_time = start_time + self.fixed_window_dt

        file_index = 0

        while(end_time < tab_last["timestamp"]):

            idx = np.searchsorted(labels["timestamp"], end_time, side="left")
            row = labels.iloc[idx]["row"]
            col = labels.iloc[idx]["col"]
            if row < 0:
                print("Row < 0: ", row)
            if col < 0:
                print("Row < 0: ", col)
            stimulus = labels.iloc[idx]["stimulus_type"]
            if stimulus == "st" or stimulus == "pa":
                old_stimulus = stimulus
                idx = find_index_with_different_stimulus(labels, idx, stimulus)
                start_time = labels["timestamp"].iloc[idx]   
                row = labels.iloc[idx]["row"]
                col = labels.iloc[idx]["col"]
                end_time = start_time + self.fixed_window_dt   
            if old_row != row and old_col != col:
                if stimulus == "s" and old_stimulus == "s":
                    state_label = int(1)
                if stimulus == "p" and old_stimulus == "p":
                    state_label = int(2)
                else:
                    state_label = int(3)
            else:
                state_label = int(0)
            old_row = row
            old_col = col
            # frame
            data_temp = np.zeros(
                (self.input_channel, self.img_width, self.img_height)
            ).astype(np.int8)
            evs_t = data["t"][(data["t"] >= start_time) & (data["t"] <= end_time)]
            evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

            np.add.at(data_temp[0], (evs_xy[evs_p == 0, 0], evs_xy[evs_p == 0, 1]), 1)
            if self.input_channel > 1:
                np.add.at(
                    data_temp[self.input_channel - 1], (evs_xy[evs_p == 1, 0], evs_xy[evs_p == 1, 1]), 1
                )
                data_temp[0, :, :][
                    data_temp[1, :, :] >= data_temp[0, :, :]
                ] = 0  # if ch 1 has more evs than 0
                data_temp[1, :, :][
                    data_temp[1, :, :] < data_temp[0, :, :]
                ] = 0  # if ch 0 has more evs than 1

            data_temp = data_temp.clip(0, 1)  # no double events
 
            batch_label = np.column_stack((row, col, state_label)).astype(np.int16)

            data_filename = f'{self.cache_data_dir}/{user_id}_{eye}_{file_index}_data.h5'
            label_filename = f'{self.cache_data_dir}/{user_id}_{eye}_{file_index}_label.h5'
            file_index += 1
            with h5py.File(data_filename, 'w') as hf:
                hf.create_dataset('data', data=data_temp, compression='gzip')

            with h5py.File(label_filename, 'w') as hf:
                hf.create_dataset('labels', data=batch_label, compression='gzip')

            with open(txt_file, 'a') as f:
                f.write(f'{data_filename}\n')
            print("Write to txt file: ", data_filename)
            print("Write to txt file: ", label_filename)
            start_time = end_time
            end_time = start_time + self.fixed_window_dt

    def load_static_window(self, data, labels):
        data = {
            "xy": np.hstack(
                [data["x"].reshape(-1, 1), data["y"].reshape(-1, 1)]
            ),
            "p": data["p"] * 1,
            "t": data["t"],
        }
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

            start_time = end_time
            end_time = start_time + self.fixed_window_dt * self.num_bins                

        return np.array(batch_data).astype(np.float32), np.array(batch_label).astype(np.float32)

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