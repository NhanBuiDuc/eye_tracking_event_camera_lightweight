from data.dataset.ini_30_dataset import Ini30Dataset
from data.dataset.DynamicDataset import DynamicDatasetHz10000
from data.dataset.hz_10000 import DatasetHz10000
import torch
import json
from model.B_3ET import Baseline_3ET
from model.ANN_retina import Retina
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer import Trainer
from torch.distributed import init_process_group, destroy_process_group
import os
from pathlib import Path
from utils.ini_30.util import get_model_for_baseline
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics.metric_base import Metric, MetricSequence
from metrics.mean_squared_error import MeanSquaredError
import torch.nn as nn
from loss.loss_base import Loss, LossSequence
from loss.YoloLoss import YoloLoss
from model.simple_convlstm import SimpleConvLSTM
from model.simple_convlstm1 import SimpleConvLSTM1
from model.simple_convlstm2 import SimpleConvLSTM2
from model.simple_convlstm3 import SimpleConvLSTM3
from model.resnet import resnet18
from metrics.AngularError import AngularError
import re
from loss.EyeGazeLoss import EyeGazeLoss
def setup_ddp(rank, world_size):
    # Set necessary environment variables
    os.environ["MASTER_ADDR"] = "localhost"  # Replace with your master node address
    os.environ["MASTER_PORT"] = "12365"     # Replace with your master node port

    # try:
    # Set world size based on the number of GPUs
    # os.environ["WORLD_SIZE"] = world_size

    # os.environ["LOCAL_RANK"] = str(rank)
    # Initialize the process group
    try:
        torch.cuda.set_device(rank)
        init_process_group(
            backend="nccl",
            # init_method="env://",
            world_size=world_size,
            rank=rank  # This process's rank, starting from 0
        )
    except Exception as e:
        print(e)
        print(f"Failed to initialize DDP: {e}")
        # Fallback to single GPU training
        torch.cuda.set_device(0)  # Use the first GPU in the list
        os.environ["LOCAL_RANK"] = "0"  # Set local rank to 0 for single GPU

def prepare_dataloader(dataset, batch_size):
    try:
        # Function to prepare DataLoader with DistributedSampler
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5,
            drop_last=True
        )
    except:
        # Function to prepare DataLoader with DistributedSampler
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5,
            drop_last=True
        )

def create_metrics_sequence(metrics: list):
    results = []
    for metric in metrics:
        if metric == "mean_squared_error":
            results.append(MeanSquaredError())
        if metric == "angular_error":
            results.append(AngularError(40))
    metrics_sequence = MetricSequence(results)
    return metrics_sequence

def create_losses_sequence(losses: list, dataset_params: dict, training_params: dict ):
    results = []
    for loss in losses:
        if loss == "yolo_loss":
            results.append(YoloLoss(dataset_params, training_params))
        if loss == "mean_squared_error":
            results.append(nn.MSELoss())
    losses_sequence = LossSequence(results)
    return losses_sequence

def load_checkpoint(snapshot_path, model, optimizer, epoch=None):
    class_name = str(type(model)).split('.')[-1][:-2]  # Extract class name (e.g., 'Retina', 'SimpleConvLSTM')
    
    # List all files in the snapshot_path directory
    all_files = os.listdir(snapshot_path)
    
    # Filter files that match the pattern "<class_name>_epoch_<number>.pt"
    pattern = rf'{class_name}_epoch_(\d+)\.pt'
    checkpoint_files = [f for f in all_files if re.match(pattern, f)]
    
    # Debugging print statement to see the files being matched
    print(f"Matching checkpoint files: {checkpoint_files}")
    
    if not checkpoint_files:
        # No checkpoints found, start from epoch 0
        print(f"No checkpoints found for {class_name}. Starting from scratch.")
        return model, optimizer, 0  
    
    if epoch is not None:
        # Load the specified epoch
        checkpoint_file = f'{class_name}_epoch_{epoch}.pt'
        if checkpoint_file not in checkpoint_files:
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found.")
    else:
        # Extract epoch numbers from file names
        epochs = [int(re.search(pattern, f).group(1)) for f in checkpoint_files]
        
        # Find the maximum epoch number
        max_epoch = max(epochs)
        
        # Construct the checkpoint file name with the maximum epoch number
        checkpoint_file = f'{class_name}_epoch_{max_epoch}.pt'
    
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(snapshot_path, checkpoint_file))
    model.load_state_dict(checkpoint["MODEL_STATE"])
    optimizer.load_state_dict(checkpoint["OPTIMIZER"])
    
    print(f"Loaded checkpoint '{checkpoint_file}' (epoch {epoch if epoch is not None else max_epoch}).")
    
    # Return the model, optimizer, and the next epoch number
    next_epoch = (epoch + 1) if epoch is not None else (max_epoch + 1)
    return model, optimizer, next_epoch

def main(train_dataset, val_dataset, test_dataset, dataset_params, training_params):

    # # Create a thread to run the load_cached_data method
    # thread = threading.Thread(target=val_dataset.load_cached_data)

    # # Start the thread

    arch_name = training_params["arch_name"]
    optimizer =  training_params["optimizer"]
    lr_model = training_params["lr_model"]
    batch_size = training_params["batch_size"]
    num_epochs = training_params["num_epochs"]
    metrics = training_params["metrics"]
    losses = training_params["losses"]
    device = training_params["device"]
    # load_checkpoint = training_params["load_checkpoint"]
    save_every = 1
    snapshot_path = "checkpoints"
    # Create a Path object
    path = Path(snapshot_path)
    Path(snapshot_path).mkdir(parents=True, exist_ok=True)

    if arch_name == "3ET":
        model = Baseline_3ET(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]
        )

    elif arch_name == "Retina":
        config = get_model_for_baseline(dataset_params, training_params)
        model = Retina(dataset_params, training_params, config)
    elif arch_name == "LSTM": 
        model = SimpleConvLSTM(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]
        )
    elif arch_name == "LSTM1": 
        model = SimpleConvLSTM1(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]
        )
    elif arch_name == "LSTM2": 
        model = SimpleConvLSTM2(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]
        )
    elif arch_name == "LSTM3": 
        model = SimpleConvLSTM3(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]
        )
    elif arch_name == "Resnet": 
        model = resnet18(num_classes = 6)
    # Initialize Optimizer
    if training_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            params= model.parameters(),
            lr=lr_model,
            weight_decay=1e-4
        )  # weight_decay=5e-5
    elif training_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(lr=lr_model, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError

    # Initialize Scheduler
    if training_params["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.8
        )
    elif training_params["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5
        )
    else:
        raise NotImplementedError
    loss = EyeGazeLoss(device = device)
    # criterions_sequence = create_losses_sequence(losses, dataset_params, training_params)
    metrics_sequence = create_metrics_sequence(metrics)
    model, optimizer, start_epoch = load_checkpoint(snapshot_path, model, optimizer)
    dataloader_list = []

    train_dataloader = prepare_dataloader(train_dataset, batch_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size)
    test_dataloader = prepare_dataloader(test_dataset, batch_size)
    dataloader_list.append(train_dataloader)
    dataloader_list.append(val_dataloader)
    dataloader_list.append(test_dataloader)
    # train_dataloader = prepare_dataloader(train_dataset, batch_size)
    trainer = Trainer(dataset_params, model.to(device) , device, dataloader_list, optimizer, scheduler, loss, metrics_sequence, save_every, snapshot_path)
    trainer.train(start_epoch, num_epochs)
    # thread.join()
    # trainer.evaluate()

if __name__ == "__main__":
    # Example function to load your dataset, model, and optimizer
    assert torch.cuda.is_available()
    torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)

    config_path = "config/evb_eye2.json"
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_params = json.load(f)
    dataset_params = config_params["dataset_params"]
    training_params = config_params["training_params"]
    short_train = False
    train_dataset = DatasetHz10000(split="train", config_params=config_params)  # Example dataset
    # val_dataset = DynamicDatasetHz10000(split="val", config_params=config_params, cache_size = 5)  # Example dataset
    # test_dataset = DatasetHz10000(split="test", config_params=config_params)  # Example dataset
    # cache = dataset_params["use_cache"]
    # if cache == False:
    #     train_dataset.prepare_unstructured_data()
    #     # val_dataset.prepare_unstructured_data()
    # else:
    #     # train_dataset.load_cached_data([1, 2])
    #     pass
    torch.set_default_device(training_params["device"])
    train_dataset.read_file_list()
    if short_train:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        # val_dataset = torch.utils.data.Subset(val_dataset, range(100))
        # test_dataset = torch.utils.data.Subset(val_dataset, range(100))

    main(train_dataset, None, None, dataset_params, training_params) 