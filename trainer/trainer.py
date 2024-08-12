from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import deque

class Trainer(ABC):
    """
    An abstract base class for distributed training across multiple GPUs.
    """

    def __init__(self,
                dataset_params,
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
        self.dataset_params = dataset_params
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
        self.num_bins = dataset_params["num_bins"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]
        self.input_channel = dataset_params["input_channel"]

    def train(self, start_epoch, max_epochs):
        best_train_loss = float('inf')  # Initialize with infinity

        for epoch in range(start_epoch, max_epochs):
            train_loss = self._run_epoch(epoch, max_epochs)
            
            # Check if the training loss has improved
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print(f"Training loss improved to {best_train_loss}")
            else:
                print(f"Training loss did not improve, stepping scheduler.")
                self.scheduler.step()

            self._save_snapshot(epoch)

    def backward(self, loss):
        loss.backward(retain_graph=True)

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

            # Save the concatenated tensor
            filename = os.path.join(directory, f"{key}.npy")
            np.save(filename, value)

            
    # def _run_epoch(self, epoch, max_epochs):
    #     self.model.train()
    #     print(f"[Epoch {epoch} | Batchsize: {self.train_data_loader.batch_size} | Steps: {len(self.train_data_loader)}")
    #     # self.train_data_loader.sampler.set_epoch(epoch)
    #     outputs = []
    #     targets = []
    #     in_output = None
    #     hidden = None
    #     source_buffer = deque(maxlen=self.num_bins)
    #     total_loss = 0
    #     # gpus = []
    #     for source, target in tqdm(self.train_data_loader):
    #         b, seq_len, c = target.shape
    #         source = source.to(self.gpu_id).float()
    #         target = target.to(self.gpu_id).float()
    #         target = target.view(b, c)
    #         self.optimizer.zero_grad()
    #         source_buffer.append(source.clone())
            
    #         # Check if the buffer is full
    #         if len(source_buffer) == self.num_bins:
    #             # Convert the deque to a tensor and reshape
    #             merged_tensor = torch.stack(list(source_buffer), dim=1)
    #             if in_output is not None:
    #                 in_output = in_output.detach()  # Detach in_output to break the computation graph
    #             # Detach hidden states to prevent them from affecting new computations
    #             if hidden is not None:
    #                 hidden = [
    #                     [(h.detach(), c.detach()) for h, c in layer_hidden_state]
    #                     for layer_hidden_state in hidden
    #                 ]
    #             output, hidden = self.model(merged_tensor, hidden, in_output)
    #             loss = self.criterions(output.float(), target.float())
    #             total_loss += loss.item()
    #             print("Train Loss: ", loss)
                
    #             in_output = output.clone()
    #             outputs.append(output.cpu().detach().numpy())
    #             targets.append(target.cpu().detach().numpy())
                
    #             self.backward(loss)
    #             self.optimizer.step()

    #     # Calculate the average loss
    #     avg_loss = total_loss / len(self.train_data_loader)
    #     # Concatenate all outputs and targets
    #     outputs = np.concatenate(outputs, axis=0)
    #     targets = np.concatenate(targets, axis=0)

    #     log_dict = {
    #         f"gpu_{self.gpu_id}_output": outputs,
    #         f"gpu_{self.gpu_id}_target": targets,
    #     }
    #     path = Path("cache/train/")
    #     self.save_numpy(log_dict, path)
    #     return avg_loss

    def _run_epoch(self, epoch, max_epochs):
        self.model.train()
        print(f"[Epoch {epoch} | Batchsize: {self.train_data_loader.batch_size} | Steps: {len(self.train_data_loader)}")
        # self.train_data_loader.sampler.set_epoch(epoch)
        outputs = []
        targets = []
        in_output = None
        hidden = None
        source_buffer = deque(maxlen=self.num_bins)
        total_loss = 0

        # # Initialize a variable to hold the accumulated tensor
        # accumulated_tensor = torch.zeros((self.num_bins, *source.shape[1:]), device=self.gpu_id)

        for source, target in tqdm(self.train_data_loader):
            b, seq_len, c = target.shape

            source = source.to(self.gpu_id).float()
            target = target.to(self.gpu_id).float()
            target = target.view(b, c)
            self.optimizer.zero_grad()

            # Add the new source to the buffer
            if len(source_buffer) == self.num_bins:
                # Remove the oldest sample from the accumulated tensor
                oldest_sample = source_buffer.popleft()
                accumulated_tensor -= oldest_sample
                
            # Add the new sample to the buffer and the accumulated tensor
            source_buffer.append(source.clone())

            if len(source_buffer) == 1:
                accumulated_tensor = source
            accumulated_tensor += source
            
            # Process the accumulated tensor when the buffer is full
            if len(source_buffer) == self.num_bins:
                # Convert the deque to a tensor and reshape
                input_tensor = torch.tensor(accumulated_tensor).to(self.gpu_id)
                if in_output is not None:
                    in_output = in_output.detach()  # Detach in_output to break the computation graph
                # Detach hidden states to prevent them from affecting new computations
                if hidden is not None:
                    hidden = [
                        [(h.detach(), c.detach()) for h, c in layer_hidden_state]
                        for layer_hidden_state in hidden
                    ]
                output, hidden = self.model(input_tensor, hidden, in_output)
                loss = self.criterions(output.float(), target.float())
                total_loss += loss.item()
                print("Train Loss: ", loss)

                in_output = output.clone()
                outputs.append(output.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())

                self.backward(loss)
                self.optimizer.step()

        # Calculate the average loss
        avg_loss = total_loss / len(self.train_data_loader)
        # Concatenate all outputs and targets
        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)

        log_dict = {
            f"gpu_{self.gpu_id}_output": outputs,
            f"gpu_{self.gpu_id}_target": targets,
        }
        path = Path("cache/train/")
        self.save_numpy(log_dict, path)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        # val_loss = 0.0

        outputs = []
        targets = []
        in_output = None
        hidden = None
        source_buffer = deque(maxlen=self.num_bins)
        with torch.no_grad():
            for source, target in tqdm(self.val_data_loader):
                b, seq_len, c = target.shape
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                target = target.view(b, c)
                source_buffer.append(source.clone())
                # Check if the buffer is full
                if len(source_buffer) == self.num_bins:
                    # Convert the deque to a tensor and reshape
                    merged_tensor = torch.stack(list(source_buffer), dim=1)
                    if in_output is not None:
                        in_output = in_output.detach()  # Detach in_output to break the computation graph
                    # Detach hidden states to prevent them from affecting new computations
                    if hidden is not None:
                        hidden = [
                            [(h.detach(), c.detach()) for h, c in layer_hidden_state]
                            for layer_hidden_state in hidden
                        ]
                    output, hidden = self.model(merged_tensor, hidden, in_output)
                    total_loss = self.criterions(output.float(), target.float())
                    print("Eval Loss: ", total_loss)
                    in_output = output.clone()
                    outputs.append(output.cpu().detach().numpy())
                    targets.append(target.cpu().detach().numpy())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        
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
        in_output = None
        hidden = None
        source_buffer = deque(maxlen=self.num_bins)
        with torch.no_grad():
            val_loss = 0
            for source, target in tqdm(self.test_data_loader):
                b, seq_len, c = target.shape
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                target = target.view(b, c)
                source_buffer.append(source.clone())
                for t in range(seq_len):

                    if t == 0:
                        out, hidden_states = self.model(data[:, t, :, :, :].view(b, 1, *data.shape[2:]), None)
                    else:
                        out, hidden_states = self.model(data[:, t, :, :, :].view(b, 1, *data.shape[2:]), hidden_states)

                    timestep_outputs.append(out)

                # Convert timestep_outputs to a tensor
                output = torch.stack(timestep_outputs)
                
                # Ensure the output tensor is contiguous before using view
                output = output.contiguous()
                
                # Use view to reshape the tensor to the desired shape
                output = output.view(target.shape)
                outputs.append(output.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy()) 
                # gpus.append(self.gpu_id)

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)

        log_dict = {
            f"gpu_{self.gpu_id}_output": outputs,
            f"gpu_{self.gpu_id}_target": targets,
        }
        path = Path("cache/test/")
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
            "OPTIMIZER": self.optimizer.state_dict()
        }
        class_name = str(type(self.model)).split('.')[-1][:-2]  # Extract class name 'Retina'
        file_name = f"{self.snapshot_path}/{class_name}_epoch_{epoch}.pt"

        torch.save(snapshot, file_name)
        print(f"Epoch {epoch} | Training snapshot saved at {file_name}")
