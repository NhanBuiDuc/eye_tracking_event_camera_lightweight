from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import numpy as np
from pathlib import Path

class Trainer(ABC):
    """
    An abstract base class for distributed training across multiple GPUs.
    """

    def __init__(self, 
                model,
                gpu_id,
                dataloader_list,
                optimizer,
                scheduler,
                criterions,
                metrics,
                save_every, 
                snapshot_path):
        """
        Initializes the DistributedTrainerBase with the given model, data loader, optimizer, criterion, and device IDs.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            data_loader (dict): A dictionary containing 'train' and 'val' DataLoader.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            criterion (torch.nn.Module): The loss function.
            device_ids (list): List of GPU device IDs to use for training.
        """
        self.gpu_id = gpu_id  # Will be set during DDP setup
        self.model = model.to(self.gpu_id)  # Move model to GPU
        self.train_data_loader = dataloader_list[0]
        self.val_data_loader = dataloader_list[1]
        self.test_data_loader = dataloader_list[2]

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterions = criterions
        self.metrics = metrics
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0

    def train(self, max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._save_snapshot(epoch)

    def backward(self, loss):
        loss.backward()

    def save_csv(self, log_dict, path):
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if path.exists():
            # If the file exists, open in append mode
            mode = 'a'
        else:
            # If the file does not exist, create it and write header
            mode = 'w'

        with open(path, mode, newline='') as csvfile:
            fieldnames = log_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if mode == 'w':
                writer.writeheader()
            writer.writerow(log_dict)
            
    def save_numpy(self, log_dict, directory):
        os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
        
        for key, value in log_dict.items():
            filename = os.path.join(directory, f"{key}.npy")
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy()
            np.save(filename, value)
            
    def _run_epoch(self, epoch):
        self.model.train()
        print(f"[Epoch {epoch} | Batchsize: {self.train_data_loader.batch_size} | Steps: {len(self.train_data_loader)}")
        # self.train_data_loader.sampler.set_epoch(epoch)
        outputs = []
        targets = []
        # gpus = []
        for source, target in (self.train_data_loader):
            b, t, c = target.shape
            timestep_outputs = []

            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self.optimizer.zero_grad()
            for idx, data in enumerate(source):
                data = data.unsqueeze(0)
                if idx == 0:
                    output, hidden_states = self.model(data, None)
                else:
                    output, hidden_states = self.model(data, hidden_states)              

                timestep_outputs.append(output)
            # Convert timestep_outputs to a tensor
            timestep_outputs_tensor = torch.stack(timestep_outputs)
            outputs.append(timestep_outputs_tensor.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
            # gpus.append(self.gpu_id)

            total_loss, loss_dict = self.criterions(output, target)

            # if self.gpu_id == 0:
            for loss in loss_dict:
                print(f"{(loss)} Train Loss: ", loss_dict[loss])

            self.backward(total_loss)
            self.optimizer.step()
        if epoch == 0:
            log_dict = {
                f"gpu_{self.gpu_id}_output": outputs,
                f"gpu_{self.gpu_id}_target": targets,
            }
            path = Path("cache/train/")
            self.save_numpy(log_dict, path)

    def evaluate(self):
        self.model.eval()
        # val_loss = 0.0

        outputs = []
        targets = []
        # gpus = []
        # num_batches = len(self.val_data_loader)
        # total_val_loss = 0.0
        # loss_dict_accumulator = {}  # To accumulate total losses by type
        with torch.no_grad():
            val_loss = 0
            for source, target, avg_dt in (self.val_data_loader):
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                output = self.model(source)

                outputs.append(output.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())
                # gpus.append(self.gpu_id)

        log_dict = {
            f"gpu_{self.gpu_id}_output": outputs,
            f"gpu_{self.gpu_id}_target": targets,
        }
        path = Path("cache/eval/")
        self.save_numpy(log_dict, path)

    def test(self):
        self.model.eval()
        # val_loss = 0.0

        outputs = []
        targets = []
        # gpus = []
        # num_batches = len(self.val_data_loader)
        # total_val_loss = 0.0
        # loss_dict_accumulator = {}  # To accumulate total losses by type
        with torch.no_grad():
            val_loss = 0
            for source, target, avg_dt in (self.test_data_loader):
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                output = self.model(source)

                outputs.append(output.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())
                # gpus.append(self.gpu_id)

        log_dict = {
            f"gpu_{self.gpu_id}_output": outputs,
            f"gpu_{self.gpu_id}_target": targets,
        }
        path = Path("cache/eval/")
        self.save_numpy(log_dict, path)
    
    def reshape(self, tensor, shape):
        """
        Reshapes a given tensor to the specified shape.

        Args:
        tensor (torch.Tensor): The tensor to reshape.
        shape (tuple): A tuple containing the desired shape.

        Returns:
        torch.Tensor: The reshaped tensor.
        """
        return tensor.reshape(*shape)
    
    def _load_snapshot(self, epoch):
        loc = f"cuda:{self.gpu_id}"
        class_name = str(type(self.model)).split('.')[-1][:-2]  # Extract class name 'Retina'
        os.makedirs(f"{self.snapshot_path}/pytorch", exist_ok=True)
        file_name = f"{self.snapshot_path}/pytorch/{class_name}_epoch_{epoch}.pt"
        snapshot = torch.load(file_name, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        class_name = str(type(self.model)).split('.')[-1][:-2]  # Extract class name 'Retina'
        file_name = f"{self.snapshot_path}/{class_name}_epoch_{epoch}.pt"

        torch.save(snapshot, file_name)
        print(f"Epoch {epoch} | Training snapshot saved at {file_name}")
