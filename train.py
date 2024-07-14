from data.dataset.ini_30_dataset import Ini30Dataset
import torch
import json
from model.B_3ET import Baseline_3ET
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer.distributed_trainer import DistributedGPUTrainer
from torch.distributed import init_process_group, destroy_process_group
import os

def setup_ddp():
    # Set necessary environment variables
    os.environ["MASTER_ADDR"] = "localhost"  # Replace with your master node address
    os.environ["MASTER_PORT"] = "12345"     # Replace with your master node port

    # Check if NNCL is available
    use_nccl = torch.cuda.nccl.is_available()

    if use_nccl:
        os.environ["RANK"] = "0"            # Replace with the rank of this process
        os.environ["WORLD_SIZE"] = "1"      # Replace with the total number of processes

        # Initialize the process group
        init_process_group(backend="nccl", init_method="env://")
        
        # Set CUDA device for this process (useful when each process uses a single GPU)
        os.environ["LOCAL_RANK"] = str(torch.distributed.get_rank())
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # Fallback to single GPU training
        torch.cuda.set_device(0)  # Use GPU 0
        os.environ["LOCAL_RANK"] = "0"  # Set local rank to 0 for single GPU

def prepare_dataloader(dataset, batch_size):
    
    # Function to prepare DataLoader with DistributedSampler
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False)
    )

if __name__ == "__main__":
    # Example function to load your dataset, model, and optimizer
    assert torch.cuda.is_available()
    torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)
    config_path = "config\ini_30.json"
    setup_ddp()
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_params = json.load(f)
    dataset_params = config_params["dataset_params"]
    training_params = config_params["training_params"]
    arch_name = "3ET"
    optimizer =  training_params["optimizer"]
    lr_model = training_params["lr_model"]
    batch_size = training_params["batch_size"]
    num_epochs = training_params["num_epochs"]
    save_every = 1
    snapshot_path = "checkpoints"

    if arch_name == "3ET":
        model = Baseline_3ET(
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

    train_dataset = Ini30Dataset(split="train", config_params=config_params)  # Example dataset
    val_dataset = Ini30Dataset(split="val", config_params=config_params)  # Example dataset
    # test_dataset = Ini30Dataset(split="test", config_json_path=config_params)  # Example dataset
    dataloader_list = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_dataloader = prepare_dataloader(train_dataset, batch_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size)
    dataloader_list.append(train_dataloader)
    dataloader_list.append(val_dataloader)
    dataloader_list.append(None)
    # train_dataloader = prepare_dataloader(train_dataset, batch_size)
    trainer = DistributedGPUTrainer(model, dataloader_list, optimizer, scheduler, save_every, snapshot_path)
    trainer.train(num_epochs)
    destroy_process_group()