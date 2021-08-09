import os

from normalize_dataset.normalized_data_loader import NormalizedDataLoader
from clean_dataset.data_cleaner import DataCleaner

base_path = "normalized_dataset"
new_path = "normalize_dataset_clean"
all_subjects = os.listdir(base_path)
outlier_method = 'iForest'

data_loader = NormalizedDataLoader(dataset_path=base_path)
data_cleaner = DataCleaner(base_path=base_path, data_loader=data_loader, outlier_method='iForest')
data_cleaner.run_and_save(subjects=all_subjects, base_folder=new_path)
