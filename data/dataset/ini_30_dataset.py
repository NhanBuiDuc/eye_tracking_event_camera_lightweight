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

from abc import ABC, abstractmethod

class GetItemStrategy(ABC):
    @abstractmethod
    def get_item(self, dataset, index):
        pass

class TonicTransformGetItemStrategy(GetItemStrategy):
    def get_item(self, dataset, transforms, index):
        
        labels = dataset.load_labels(index)
        events = dataset.load_events(index)
        tmp_struct = make_structured_array(
            events["xy"][:, 0], events["xy"][:, 1], events["t"], events["p"]
        )
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
    def get_item(self, dataset, transforms, index):
        
        labels = dataset.load_labels(index)
        events = dataset.load_events(index)
        tmp_struct = make_structured_array(
            events["xy"][:, 0], events["xy"][:, 1], events["t"], events["p"]
        )
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
            "p": tmp_struct["p"] * 1,
            "t": tmp_struct["t"],
        }
        events, labels = dataset.load_static_window(events, labels)
        event_tensor = events.float()
        labels_tensor = labels.float()
        return event_tensor, labels_tensor, dataset.fixed_window_dt

class DynamicWindowGetItemStrategy(GetItemStrategy):
    def get_item(self, dataset, transforms, index):
        labels = dataset.load_labels(index)
        events = dataset.load_events(index)
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
        event_tensor = events.float()
        labels_tensor = labels.float()
        return event_tensor, labels_tensor, avg_dt


class Ini30Dataset:
    def __init__(
        self,
        split: str,
        config_params: dict,
    ):
        self.split = split
        # self.shuffle = shuffle
        for key, value in config_params.items():
            setattr(self, key, value)
            for sub_key, sub_value in value.items():
                setattr(self, sub_key, sub_value)

        if self.get_item_strategy == "static_window":
            self.get_item_strategy = StaticWindowGetItemStrategy()
        elif self.get_item_strategy == "dynamic_window":
            self.get_item_strategy = DynamicWindowGetItemStrategy()

        self.train_idxs, self.val_idxs = get_indexes(val_idx=self.val_idx)
        self.transform, self.target_transform = get_transforms(self.dataset_params, self.training_params)
        if self.split == "train":      
            self.list_experiments = self.train_idxs
        else:
            self.list_experiments = self.val_idxs

        self.y = pd.read_csv(os.path.join(self.data_dir, "silver.csv"), delimiter='\t')
        # Standardize column names: strip spaces and convert to lowercase
        self.y.columns = self.y.columns.str.strip().str.lower()
        print(self.y.columns)
        self.experiments = np.unique(self.y["exp_name"]).tolist()

        filter_values = [self.experiments[item] for item in self.list_experiments]
        self.y = self.y[self.y["exp_name"].isin(filter_values)]

        # correct cropped
        """
        Since the algorithm is deployed on the neuromorphic SoC
        Speck, which has two-channel support for a 64x64 DVS’s
        resolution, we prepared the dataset to bridge this domain
        gap. First, we transformed the rectangular resolution of the
        original data to a squared array of 512x512, by shifting the
        y-axis to 16 pixels and discarding 128 X-coordinates in cor
        respondence to the spatial location where fewer events are
        present and no label appeared (x < 96 and x > 608).
        """
        self.min_x, self.max_x = 96, 608
        self.y = self.y[(self.y.x_coord > self.min_x) & (self.y.x_coord < self.max_x)]
        # shifting the range [96, 608] to [0, 512]
        self.y.x_coord -= self.min_x
        # The y-coordinates are shifted by adding 16 pixels. 
        self.y.y_coord += 16

        self.avg_dt = 0
        self.items = 0

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, index):
        return self.get_item_strategy.get_item(self, self.tonic_transforms, index)
    
    def load_labels(self, index):
        # collect labels
        item = self.y.iloc[index]
        path_to_exp = os.path.join(self.data_dir, item["exp_name"])

        tab = read_csv(
            pathlib.Path(os.path.join(path_to_exp, "annotations.csv")), False, True
        )
        tab = tab.sort_values(by="timestamp")
        tab = tab[tab["timestamp"] <= item["t_end"]]
        if self.get_item_strategy == "static_window":
            tab = tab[
                tab["timestamp"]
                >= item["t_end"] - self.fixed_window_dt * (self.num_bins + 1)
            ]

        # center crop labels : 640x480 -> 512x512
        tab.center_x = 512 - (tab.center_x - self.min_x)
        tab.center_y = 512 - (tab.center_y + 16)

        return tab

    def load_events(self, index):
        # collect events
        item = self.y.iloc[index]
        path_to_exp = os.path.join(self.data_dir, item["exp_name"])
        aedat_path = pathlib.Path(os.path.join(path_to_exp, "events.aedat4"))
        aedat_processor = AedatProcessorLinear(aedat_path, 0.25, 1e-7, 0.5)  # 1e-7
        events = aedat_processor.collect_events(0, item["t_end"])
        evs_coord = events.coordinates()
        evs_timestamp = events.timestamps()
        evs_features = events.polarities().astype(np.byte)

        # center crop events : 640x480 -> 512x512
        evs_idx = (evs_coord[:, 0] > self.min_x) & (evs_coord[:, 0] < self.max_x)
        evs_timestamp = evs_timestamp[evs_idx]
        evs_features = evs_features[evs_idx]
        evs_coord = evs_coord[evs_idx, :]
        evs_coord[:, 0] -= self.min_x
        evs_coord[:, 1] += 16

        # down sample : 512x512 -> img_width x img_height
        evs_coord //= 512 // self.img_width

        return {"t": evs_timestamp, "p": evs_features, "xy": evs_coord}
    
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
    
    def load_static_window(self, data, labels):
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        start_label = (int(tab_start.center_x.item()), int(tab_start.center_y.item()))
        end_label = (int(tab_last.center_x.item()), int(tab_last.center_y.item()))

        start_time = tab_last["timestamp"] - self.fixed_window_dt * (self.num_bins + 1)
        evs_t = data["t"][data["t"] >= start_time]
        evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

        # frame
        data = np.zeros(
            (self.num_bins, self.input_channel, self.img_width, self.img_height)
        )

        # indexes
        start_idx = 0

        # get intermediary labels based on num of bins
        fixed_timestamps = np.linspace(start_time, tab_last["timestamp"], self.num_bins)
        x_axis, y_axis = [], []

        for i, fixed_tmp in enumerate(fixed_timestamps):
            # label
            idx = np.searchsorted(labels["timestamp"], fixed_tmp, side="left")
            if idx == 0:
                x_axis.append(start_label[0])
                y_axis.append(start_label[1])
            elif idx == len(labels["timestamp"]):
                x_axis.append(end_label[0])
                y_axis.append(end_label[1])
            else:  # Weighted interpolation
                t0 = labels["timestamp"].iloc[idx - 1]
                t1 = labels["timestamp"].iloc[idx]

                weight0 = (t1 - fixed_tmp) / (t1 - t0)
                weight1 = (fixed_tmp - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_x"] * weight0
                        + labels.iloc[idx]["center_x"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_y"] * weight0
                        + labels.iloc[idx]["center_y"] * weight1
                    )
                )

            # slice
            t = evs_t[start_idx:][evs_t[start_idx:] <= fixed_tmp]
            if t.shape[0] == 0:
                continue
            xy = evs_xy[start_idx : start_idx + t.shape[0], :]
            p = evs_p[start_idx : start_idx + t.shape[0]]

            np.add.at(data[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
            if self.input_channel > 1:
                np.add.at(
                    data[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                )
                data[i, 0, :, :][
                    data[i, 1, :, :] >= data[i, 0, :, :]
                ] = 0  # if ch 1 has more evs than 0
                data[i, 1, :, :][
                    data[i, 1, :, :] < data[i, 0, :, :]
                ] = 0  # if ch 0 has more evs than 1

            data[i] = data[i].clip(0, 1)  # no double events

            # move pointers
            start_idx += t.shape[0]

        frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        frames = frames.permute(0, 1, 3, 2) 
        labels = self.target_transform(np.vstack([x_axis, y_axis]))

        self.avg_dt += (evs_t[-1] - evs_t[0]) / self.num_bins
        self.items += 1

        return frames, labels

    def load_dynamic_window(self, data, labels):
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        start_label = (int(tab_start.center_x.item()), int(tab_start.center_y.item()))
        end_label = (int(tab_last.center_x.item()), int(tab_last.center_y.item()))

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
                        labels.iloc[idx - 1]["center_x"] * weight0
                        + labels.iloc[idx]["center_x"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_y"] * weight0
                        + labels.iloc[idx]["center_y"] * weight1
                    )
                )
        frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        frames = frames.permute(0, 1, 3, 2)
        x_axis.reverse()
        y_axis.reverse()
        labels = self.target_transform(np.vstack([x_axis, y_axis]))
        avg_dt = (evs_t[-1] - evs_t[start_idx + len(xy)]) / self.num_bins

        return frames, labels, avg_dt
