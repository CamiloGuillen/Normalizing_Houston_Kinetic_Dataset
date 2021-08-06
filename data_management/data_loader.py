import os
import numpy as np

from data_management.trial_loader import TrialDataLoader, TrialLabelsLoader
from tqdm import tqdm


class DataLoader:
    def __init__(self, base_path, data_folder, labels_folder, subjects):
        self.data_path = os.path.join(base_path, data_folder)
        self.labels_path = os.path.join(base_path, labels_folder)
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        files = [(trial, label) for trial, label in zip(os.listdir(self.data_path), os.listdir(self.labels_path))
                 if (str(trial).split('T')[0] and str(label).split('-')[0]) == subject]

        right_ankle_data, right_hip_data, right_knee_data = list(), list(), list()
        left_ankle_data, left_hip_data, left_knee_data = list(), list(), list()
        labels = list()

        for trial, label in tqdm(files):
            trial_data_loader = TrialDataLoader(data_path=os.path.join(self.data_path, trial))
            right_knee_data.append(trial_data_loader.joint_angle('jRightKnee'))
            left_knee_data.append(trial_data_loader.joint_angle('jLeftKnee'))
            right_ankle_data.append(trial_data_loader.joint_angle('jRightAnkle'))
            left_ankle_data.append(trial_data_loader.joint_angle('jLeftAnkle'))
            right_hip_data.append(trial_data_loader.joint_angle('jRightHip'))
            left_hip_data.append(trial_data_loader.joint_angle('jLeftHip'))

            trial_label_loader = TrialLabelsLoader(label_path=os.path.join(self.labels_path, label),
                                                   length_data=len(trial_data_loader.joint_angle('jRightKnee')))
            labels.append(trial_label_loader.get_labels())

        data = dict()
        data['Right_Knee'] = np.vstack(right_knee_data)
        data['Left_Knee'] = np.vstack(left_knee_data)
        data['Right_Ankle'] = np.vstack(right_ankle_data)
        data['Left_Ankle'] = np.vstack(left_ankle_data)
        data['Right_Hip'] = np.vstack(right_hip_data)
        data['Left_Hip'] = np.vstack(left_hip_data)
        data['Labels'] = np.vstack(labels)

        return data
