from tonic.transforms import Compose, ToFrame, MergePolarities, EventDrop, RandomFlipPolarity, Decimation, Denoise, CenterCrop, Downsample
from typing import List, Tuple, Union
from utils.ini_30.transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP, Downscale
import torch.nn as nn

def get_transforms(dataset_params, training_params, augmentations: bool = False) -> Tuple[Compose, FromPupilCenterToBoundingBox]:
    """
    Get input and target transforms for the Ini30Dataset.

    Parameters:
        num_bins (int): The number of bins used for transformation.
        augmentations (bool, optional): If True, apply data augmentations. Defaults to False.

    Returns:
        Tuple[Compose, FromPupilCenterToBoundingBox]: A tuple containing the input and target transforms.
    """
    sensor_size = (dataset_params["img_height"], dataset_params["img_width"], dataset_params["input_channel"])

    target_transforms = FromPupilCenterToBoundingBox(   yolo_loss=training_params["yolo_loss"], 
                                                        focal_loss=training_params["focal_loss"],
                                                        bbox_w=training_params["bbox_w"],
                                                        SxS_Grid=training_params["SxS_Grid"],
                                                        num_classes=training_params["num_classes"],
                                                        num_boxes=training_params["num_boxes"],
                                                        dataset_name=dataset_params["dataset_name"],
                                                        image_size=(dataset_params["img_width"], dataset_params["img_height"]),
                                                        num_bins=dataset_params["num_bins"]) 

    input_transforms = [AedatEventsToXYTP()]
    if dataset_params["dataset_name"]=="ini-30"and (dataset_params["img_width"] != 640 or dataset_params["img_height"] != 480):
        input_transforms.append(CenterCrop(sensor_size=(640, 480), size=(512, 512))) 
        input_transforms.append(Downscale())

    if dataset_params["pre_decimate"]:
        input_transforms.append(Decimation(dataset_params["pre_decimate_factor"]))
        
    if dataset_params["denoise_evs"]:
        input_transforms.append(Denoise(filter_time=dataset_params["filter_time"]))
    
    if dataset_params["random_flip"]:
        input_transforms.append(RandomFlipPolarity())

    if dataset_params["event_drop"]:
        input_transforms.append(EventDrop(sensor_size=sensor_size))
        
    if dataset_params["input_channel"] == 1:
        input_transforms.append(MergePolarities())
    
    input_transforms.append(ToFrame(sensor_size=sensor_size,   
                                    n_event_bins=dataset_params["num_bins"]
                                    ))
    input_transforms = Compose(input_transforms)
    return input_transforms, target_transforms

def get_indexes(val_idx): 
    train_val_idxs = list(range(0, 30))
    #random.shuffle(train_val_idxs)
    train_val_idxs.remove(val_idx)
    train_idxs = train_val_idxs
    val_idxs = [val_idx]   
    return train_idxs, val_idxs
def compute_output_dim(training_params):
    # select loss
    if training_params["yolo_loss"]:
        output_dim  = training_params["SxS_Grid"]  * training_params["SxS_Grid"] \
            *(training_params["num_classes"] + training_params["num_boxes"] * 5)
    elif training_params["focal_loss"]:
        output_dim = 4
    else:
        output_dim  = 2
    return output_dim
def get_model_for_baseline(dataset_params, training_params):
    output_dim = compute_output_dim(training_params)
    layers_config = [
        # Layer 0
        {
            "name": "Input",
            "img_width": dataset_params["img_width"],
            "img_height": dataset_params["img_height"],
            "input_channel": dataset_params["input_channel"],
        },
        # {"name": "Decimation", "decimation_rate": training_params["decimation_rate"]},

        # Layer 1
        {"name": "Conv", "out_dim": 16, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"},  # ReLU activation instead of IAF

        # Layer 2
        {"name": "Conv", "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"},  # ReLU activation instead of IAF

        # Layer 3
        {"name": "Conv", "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"}, # ReLU activation instead of IAF

        # Layer 4
        {"name": "Conv", "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"}, # ReLU activation instead of IAF

        # Layer 5
        {"name": "Conv", "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"}, # ReLU activation instead of IAF

        # Layer 6
        {"name": "Conv", "out_dim": 256, "k_xy": 3, "s_xy": 2, "p_xy": 1},
        {"name": "BatchNorm"},
        {"name": "ReLu"}, # ReLU activation instead of IAF

        # Layer 7
        {"name": "Flat"},
        {"name": "Linear", "out_dim": 512},
        {"name": "ReLu"}, # ReLU activation instead of IAF

        # Layer 8
        {"name": "Linear", "out_dim": output_dim},
    ]

    return layers_config
    
def get_summary(model):
    """
    Prints model memory
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_kb = ((param_size + buffer_size) / 1024**2)*1000
    print('Model size: {:.1f}KB'.format(size_all_kb))

    # Count the number of layers
    num_layers = sum(1 for _ in model.modules()) - 1
    print(f"Number of layers: {num_layers}")

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}") 