import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.ini_30.ini_30_aeadat_processor import AedatProcessorLinear, read_csv

def visualize(exp_folder='data/dataset/evs_ini30/ID_001'):
    # Ensure the 'images' folder exists, create it if necessary
    output_folder = os.path.join("visualization/evs_ini30",'images')
    os.makedirs(output_folder, exist_ok=True)

    # Path to annotation and events files
    annotation_file = os.path.join(exp_folder, 'annotations.csv')
    events_file = os.path.join(exp_folder, 'events.aedat4')

    # Load annotations
    annotations = read_csv(annotation_file, is_with_ellipsis=True, is_with_coords=True)

    # Load events
    aedat_processor = AedatProcessorLinear(events_file, contribution=0.25, decay=1e-7, neutral_val=0.5)
    events = aedat_processor.collect_events(0, annotations['timestamp'].max())

    # Prepare data for visualization
    events_xy = events.coordinates()
    events_polarity = events.polarities()
    
    # Print information about events_xy
    print(f"events_xy minimum: {np.min(events_xy, axis=0)}")
    print(f"events_xy maximum: {np.max(events_xy, axis=0)}")

    # Print information about events_polarity
    unique, counts = np.unique(events_polarity, return_counts=True)
    polarity_info = dict(zip(unique, counts))
    print(f"Number of 0s: {polarity_info.get(0, 0)}")
    print(f"Number of 1s: {polarity_info.get(1, 0)}")
    
    # Plot events
    plt.figure(figsize=(10, 6))
    plt.scatter(events_xy[:, 0], events_xy[:, 1], c=events_polarity, cmap='coolwarm', s=1)
    plt.title(f'Events for {exp_folder}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(os.path.join(output_folder, f'{os.path.basename(exp_folder)}_events.png'))
    plt.close()

    # Plot annotations
    plt.figure(figsize=(10, 6))
    plt.scatter(events_xy[:, 0], events_xy[:, 1], c=events_polarity, cmap='coolwarm', s=1)
    plt.scatter(annotations['center_x'], annotations['center_y'], color='red', marker='x', label='Label')
    plt.title(f'Annotations for {exp_folder}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{os.path.basename(exp_folder)}_annotations.png'))
    plt.close()

# # Example usage: Visualize 'ID_001' experiment and save the plots separately
# visualize('data/dataset/evs_ini30/ID_001')
