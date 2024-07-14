import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from trainer.distributed_trainer_base import DistributedTrainerBase
import torch.functional as F

class DistributedGPUTrainer(DistributedTrainerBase):
    def __init__(self, 
                model,
                dataloader_list,
                optimizer, 
                criterion,
                scheduler,
                save_every, 
                snapshot_path
                ):
        super().__init__(model, dataloader_list, optimizer, criterion, scheduler, save_every, snapshot_path)

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_data_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data_loader)}")
        self.train_data_loader.sampler.set_epoch(epoch)

        for source, targets in self.train_data_loader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_loss = 0
            for source, targets in self.val_data_loader:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                loss = self.criterion(output, targets)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = 100.0 * correct / total
        print(f"Validation Loss: {val_loss / len(self.val_data_loader)} | Accuracy: {accuracy}%")

    def save_model(self, path):
        torch.save(self.model.module.state_dict(), path)  # Save the model state_dict of the DDP wrapper
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.module.load_state_dict(torch.load(path))  # Load the model state_dict into the DDP wrapper
        print(f"Model loaded from {path}")
