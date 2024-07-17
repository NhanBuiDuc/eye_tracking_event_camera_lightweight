from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Abstract base class for defining loss functions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the loss function with the given parameters.
        
        Parameters:
            **kwargs: Arbitrary keyword arguments for loss configuration.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def forward(self, outputs, labels):
        """
        Abstract method to compute the loss value.

        Parameters:
            outputs (tensor): Model outputs.
            labels (tensor): Target labels.

        Returns:
            float: Loss value.
        """
        pass

class LossSequence:
    """
    Class to handle a sequence of loss functions.
    """

    def __init__(self, losses):
        """
        Initialize with a list of Loss objects.

        Parameters:
            losses (list): List of Loss objects.
        """
        self.losses = losses

    def __call__(self, outputs, labels):
        """
        Compute losses for the given outputs and labels.

        Parameters:
            outputs (tensor): Model outputs.
            labels (tensor): Target labels.

        Returns:
            dict: Dictionary mapping loss names to loss values.
        """
        loss_results = {}
        total_loss = 0.0
        for loss in self.losses:
            loss_name = type(loss).__name__  # Use class name as loss name
            loss_value = loss(outputs, labels)
            loss_results[loss_name] = loss_value
            total_loss += loss_value
        return total_loss, loss_results

    def add(self, loss: Loss):
        """
        Add a Loss object to the list of losses.

        Parameters:
            loss (Loss): Loss object to add.
        """
        self.losses.append(loss)
    def to_csv(self, path):
        """
        Export the loss results to a CSV file.

        Parameters:
            path (str): Name of the CSV file to create.
        """
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['Loss Name', 'Loss Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for loss in self.losses:
                loss_name = type(loss).__name__
                loss_value = loss(outputs, labels)  # Assuming outputs and labels are defined somewhere
                writer.writerow({'Loss Name': loss_name, 'Loss Value': loss_value})

        print(f"Loss values exported to {path} successfully.")