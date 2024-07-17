from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from pathlib import Path
import csv

class Metric(ABC, torch.nn.Module):
    """
    Abstract base class for defining metrics.
    """

    def __init__(self, **kwargs):
        """
        Initialize the metric with the given parameters.
        
        Parameters:
            **kwargs: Arbitrary keyword arguments for metric configuration.
        """
        super(Metric, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def forward(self, inputs, targets):
        """
        Abstract method to compute the metric value.

        Parameters:
            inputs (dict): Dictionary mapping metric names to input tensors.
            targets (dict): Dictionary mapping metric names to target tensors.

        Returns:
            float: Metric value.
        """
        pass

    def compute(self, inputs, targets):
        """
        Compute the metric value by calling the forward method.
        This method is provided to integrate with PyTorch's nn.Module interface.

        Parameters:
            inputs (dict): Dictionary mapping metric names to input tensors.
            targets (dict): Dictionary mapping metric names to target tensors.

        Returns:
            float: Metric value.
        """
        return self.forward(inputs, targets)
        
class MetricSequence:
    """
    Class to handle a sequence of metrics.
    """

    def __init__(self, metrics):
        """
        Initialize with a list of Metric objects.

        Parameters:
            metrics (list): List of Metric objects.
        """
        self.metrics = metrics

    def __call__(self, inputs, targets):
        """
        Compute metrics for the given inputs and targets.

        Parameters:
            inputs (dict): Dictionary mapping metric names to input tensors.
            targets (dict): Dictionary mapping metric names to target tensors.

        Returns:
            dict: Dictionary mapping metric names to metric values.
        """
        metric_results = {}
        for metric in self.metrics:
            metric_name = type(metric).__name__  # Use class name as metric name
            metric_value = metric(inputs, targets)
            metric_results[metric_name] = metric_value
        self.metric_results = metric_results
        return metric_results

    def add(self, metric: Metric):
        self.metrics.append(metric)

    def to_csv(self, path, keys_to_log=None):
        """
        Export the specified metric results to a CSV file.

        Parameters:
            path (str): Name of the CSV file to create or append to.
            keys_to_log (list, optional): List of keys to log. If None, log all keys.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if path.exists():
            # If the file exists, open in append mode
            mode = 'a'
        else:
            # If the file does not exist, create it and write header
            mode = 'w'

        with open(path, mode, newline='') as csvfile:
            fieldnames = ['Metric Name', 'Metric Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if mode == 'w':
                writer.writeheader()

            for metric, metric_value in self.metric_results.items():
                if isinstance(metric_value, dict):
                    for key, value in metric_value.items():
                        if keys_to_log is None or key in keys_to_log:
                            writer.writerow({'Metric Name': f"{metric} - {key}", 'Metric Value': value})
                else:
                    if keys_to_log is None or metric in keys_to_log:
                        writer.writerow({'Metric Name': metric, 'Metric Value': metric_value})

        print(f"Metric values exported to {path} successfully.")