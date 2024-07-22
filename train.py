from data.dataset.ini_30_dataset import Ini30Dataset
import torch
import json
from model.B_3ET import Baseline_3ET
from model.ANN_retina import Retina
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer.distributed_trainer_base import DistributedTrainerBase
from torch.distributed import init_process_group, destroy_process_group
import os
from pathlib import Path
from utils.ini_30.util import get_model_for_baseline
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics.metric_base import Metric, MetricSequence
from metrics.mean_squared_error import MeanSquaredError

from loss.loss_base import Loss, LossSequence
from loss.YoloLoss import YoloLoss
from model.simple_convlstm import SimpleConvLSTM
from data.dataset.hz_10000 import DatasetHz10000

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
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset, shuffle=False)
        )
    except:
        # Function to prepare DataLoader with DistributedSampler
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )

def create_metrics_sequence(metrics: list):
    results = []
    for metric in metrics:
        if metric == "mean_squared_error":
            results.append(MeanSquaredError())
    metrics_sequence = MetricSequence(results)
    return metrics_sequence

def create_losses_sequence(losses: list, dataset_params: dict, training_params: dict ):
    results = []
    for loss in losses:
        if loss == "yolo_loss":
            results.append(YoloLoss(dataset_params, training_params))
    losses_sequence = LossSequence(results)
    return losses_sequence

def distributed_job(rank, world_size, train_dataset, val_dataset, test_dataset, dataset_params, training_params):
    setup_ddp(rank, world_size)

    arch_name = "LSTM"
    optimizer =  training_params["optimizer"]
    lr_model = training_params["lr_model"]
    batch_size = training_params["batch_size"]
    num_epochs = training_params["num_epochs"]
    metrics = training_params["metrics"]
    losses = training_params["losses"]
    save_every = 1
    snapshot_path = "checkpoints"
    # Create a Path object
    path = Path(snapshot_path)
    path.mkdir(parents=True, exist_ok=True)

    
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
    # Initialize Optimizer
    if training_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            params= model.parameters(),
            lr=lr_model
        )  # weight_decay=5e-5
    elif training_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(lr=lr_model, momentum=0.9)
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

    criterions_sequence = create_losses_sequence(losses, dataset_params, training_params)
    metrics_sequence = create_metrics_sequence(metrics)

    # test_dataset = Ini30Dataset(split="test", config_json_path=config_params)  # Example dataset
    dataloader_list = []

    train_dataloader = prepare_dataloader(train_dataset, batch_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size)
    test_dataloader = prepare_dataloader(test_dataset, batch_size)
    dataloader_list.append(train_dataloader)
    dataloader_list.append(val_dataloader)
    dataloader_list.append(test_dataloader)
    # train_dataloader = prepare_dataloader(train_dataset, batch_size)
    trainer = DistributedTrainerBase(model, rank, dataloader_list, optimizer, scheduler, criterions_sequence, metrics_sequence, save_every, snapshot_path)
    trainer.train(num_epochs)
    trainer.evaluate()
    trainer.test()
    destroy_process_group()

if __name__ == "__main__":
    # Example function to load your dataset, model, and optimizer
    assert torch.cuda.is_available()
    torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    gpus_list = [2, 3, 4]
    n_gpus = torch.cuda.device_count()
    # print(f"Start running basic DDP example on rank {rank}.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus_list)) # Use GPUs 0 and 1
    print(f"Number of avaible GPUS: {n_gpus}.")
    print(f"Use GPUS: {len(gpus_list)}.")
    # world_size = torch.cuda.device_count()
    world_size = len(gpus_list)
    # setup_ddp(gpu_indices = [0, 1, 2], rank = rank, world_size)
    # create model and move it to GPU with id rank
    short_train = False
    config_path = "config/evb_eye.json"
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_params = json.load(f)
    dataset_params = config_params["dataset_params"]
    training_params = config_params["training_params"]
    train_dataset = DatasetHz10000(split="train", config_params=config_params)  # Example dataset
    val_dataset = DatasetHz10000(split="val", config_params=config_params)  # Example dataset
    test_dataset = DatasetHz10000(split="test", config_params=config_params)  # Example dataset
    
    if dataset_params["use_cache"] == True:
        train_dataset.load_data()
        val_dataset.load_data()
        test_dataset.load_data()

    elif dataset_params["use_cache"] == False:
        train_dataset.parallel_process_data()

    if short_train:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(100))
        test_dataset = torch.utils.data.Subset(val_dataset, range(100))

    mp.spawn(distributed_job, args=(world_size, train_dataset, val_dataset, test_dataset, dataset_params, training_params), nprocs=world_size)