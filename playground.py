import os
import numpy as np

from data_management.data_loader import DataLoader
from new_dataset.normalize_dataset import NormalizeDataset

base_path = "C:/Users/Camilo Guillen/Documents/Universidad de los Andes/Tesis/Datasets/houston/UH Dataset/"
data_folder = 'kin_data'
labels_folder = 'labels'
subjects = list(np.unique([sub.split('T')[0] for sub in os.listdir(os.path.join(base_path, data_folder))]))
key = 'hs'
samples = 100

data_loader = DataLoader(base_path=base_path, data_folder=data_folder, labels_folder=labels_folder, subjects=subjects)
normalize_dataset_obj = NormalizeDataset(data_loader=data_loader, gc_key=key, samples=samples)
normalize_dataset_obj.run()
