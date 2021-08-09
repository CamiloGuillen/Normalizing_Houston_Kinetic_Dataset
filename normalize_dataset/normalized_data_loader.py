import os
import numpy as np

from sklearn.utils import shuffle


class NormalizedDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_data(self, subject, joint, activity, shuffle_data=False):
        data = list()
        labels = list()

        b_path = os.path.join(self.dataset_path, subject, joint)
        files = os.listdir(b_path)

        if f"{activity}.npy" in files:
            file_data = np.load(f"{b_path}/{activity}.npy")
            data.append(file_data)
            labels.append(np.array([[subject, joint, activity]]*file_data.shape[0]))

        x = np.concatenate(data)
        y = np.concatenate(labels)

        if shuffle_data:
            x, y = shuffle(x, y)

        return x, y
