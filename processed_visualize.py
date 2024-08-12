import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


# Visualize the accumulated data
def visualize_data(data):
    # Data shape is (num_channels, height, width)
    num_channels, height, width = data.shape
    
    # Create an empty image to visualize
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Channel 1: Red color
    image[(data[0] == 1)] = [255, 0, 0]  # Red
    
    # Channel 2: Blue color
    image[(data[1] == 1)] = [0, 0, 255]  # Blue
    
    plt.imshow(image)
    plt.title('Accumulated 64x64 Grid Visualization')
    plt.axis('off')  # Hide axes
    plt.show()


class DataQueue:
    def __init__(self, queue_size, shape):
        self.queue_size = queue_size
        self.shape = shape
        self.queue = []
        self.current_sum = np.zeros(shape, dtype=np.float32)

    def add_sample(self, sample):
        if len(self.queue) >= self.queue_size:
            # Remove the oldest sample from the sum
            oldest_sample = self.queue.pop(0)
            self.current_sum -= oldest_sample
        
        # Add the new sample to the queue and the sum
        self.queue.append(sample)
        self.current_sum += sample

    def get_accumulated_data(self):
        return np.clip(self.current_sum, 0, 1)

# Path to the folder containing HDF5 files
folder_path = 'cache/stream_data'

# Initialize the data queue
queue_size = 5
shape = (2, 64, 64)
data_queue = DataQueue(queue_size, shape)

# List to store data from the first 20 .h5 files
file_counter = 0

# Read the first 20 .h5 files
for filename in os.listdir(folder_path):
    if filename.endswith('_data.h5'):
        file_path = os.path.join(folder_path, filename)
        print(f"Reading file: {filename}")
        
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as file:
            dataset = file['data']
            data = dataset[:]
            
            # Add the sample to the queue and update accumulated data
            data_queue.add_sample(data)
            
            file_counter += 1

        # Get the accumulated data
        accumulated_data = data_queue.get_accumulated_data()
        # Visualize the accumulated data
        visualize_data(accumulated_data)
