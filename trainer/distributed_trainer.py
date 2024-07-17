import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from trainer.distributed_trainer_base import DistributedTrainerBase
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

class DistributedGPUTrainer(DistributedTrainerBase):
    def __init__(self, 
                model,
                gpu_id,
                dataloader_list,
                optimizer, 
                scheduler,
                criterions,
                metrics,
                save_every, 
                snapshot_path
                ):
        super().__init__(model, gpu_id, dataloader_list, optimizer, scheduler, criterions, metrics, save_every, snapshot_path)

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_data_loader))[0])
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data_loader)}")
        self.train_data_loader.sampler.set_epoch(epoch)
        outputs = []
        targets = []
        # gpus = []
        for source, target, avg_dt in (self.train_data_loader):
            b, t, c, w, h = target.shape
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(source)
            output = self.reshape(output, [b, t, c, w, h])
            outputs.append(output.cpu().detach().numpy())
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
                # total_loss, loss_dict = self.criterions(output, target)
                # total_val_loss += total_loss.item()
                # # Accumulate losses from loss_dict
                # for loss_name, loss_value in loss_dict.items():
                #     if loss_name in loss_dict_accumulator:
                #         loss_dict_accumulator[loss_name] += loss_value.item()
                #     else:
                #         loss_dict_accumulator[loss_name] = loss_value.item()

        # if self.gpu_id == 0:      
        #     print(f"Average Val loss: {total_val_loss / num_batches}")
        #     for loss_name, accumulated_loss in loss_dict_accumulator.items():
        #         average_loss = accumulated_loss / num_batches
        #         print(f"{loss_name} Val Loss: {average_loss}")

        # self.metrics(outputs, targets)
        # self.metrics.to_csv(path = "log/eval.csv")
    
    def save_model(self, path):
        torch.save(self.model.module.state_dict(), path)  # Save the model state_dict of the DDP wrapper
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.module.load_state_dict(torch.load(path))  # Load the model state_dict into the DDP wrapper
        print(f"Model loaded from {path}")

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