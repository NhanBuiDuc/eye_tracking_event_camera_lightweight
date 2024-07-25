import json
from data.dataset.hz_10000 import DatasetHz10000

if __name__ == "__main__":

    config_path = "config/evb_eye.json"
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_params = json.load(f)
    dataset_params = config_params["dataset_params"]
    training_params = config_params["training_params"]
    train_dataset = DatasetHz10000(split="train", config_params=config_params)  # Example dataset
    val_dataset = DatasetHz10000(split="val", config_params=config_params)  # Example dataset
    index = [20, 21, 24]
    train_dataset.prepare_unstructured_data(index)
    # val_dataset.prepare_unstructured_data()