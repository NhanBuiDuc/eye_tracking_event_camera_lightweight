from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainerBase(ABC):
    """
    An abstract base class for distributed training across multiple GPUs.
    """

    def __init__(self, 
                model, 
                dataloader_list,
                optimizer,
                scheduler,
                criterions,
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
        self.model = model
        self.train_data_loader = dataloader_list[0]
        self.val_data_loader = dataloader_list[1]
        self.test_data_loader = dataloader_list[2]

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterions = criterions
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        self.gpu_id = None  # Will be set during DDP setup

        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    @abstractmethod
    def _run_epoch(self, epoch):
        raise NotImplementedError("Subclasses should implement _run_epoch method")
      
    # @abstractmethod
    # def train_one_epoch(self):
    #     """
    #     Trains the model for one epoch using multiple GPUs.
    #     This method should be implemented by subclasses.
    #     """
    #     pass

    # @abstractmethod
    # def evaluate(self):
    #     """
    #     Evaluates the model on the validation set using multiple GPUs.
    #     This method should be implemented by subclasses.
    #     """
    #     pass

    # @abstractmethod
    # def save_model(self, path):
    #     """
    #     Saves the model to the specified path.
        
    #     Parameters:
    #         path (str): The path where the model will be saved.
    #     """
    #     pass

    # @abstractmethod
    # def load_model(self, path):
    #     """
    #     Loads the model from the specified path.
        
    #     Parameters:
    #         path (str): The path from where the model will be loaded.
    #     """
    #     pass