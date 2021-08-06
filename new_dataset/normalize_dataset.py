import os
import numpy as np

from scipy.interpolate import interp1d


class NormalizeDataset:
    def __init__(self, data_loader, gc_key='hs', samples=50, path="normalized_dataset"):
        self.data_loader = data_loader
        self.gc_key = gc_key
        self.samples = samples
        self.path = path

    @staticmethod
    def remove_nones(data):
        none_idx = [i for i, label in enumerate(data['Labels'][:, 0]) if label == 'none']

        new_data = dict()
        for key, value in data.items():
            new_data[key] = np.delete(data[key], none_idx, axis=0)

        return new_data

    @staticmethod
    def get_stride_idx(gc_event_label, gc_key):
        stride_idx = list()
        if gc_event_label[0] == gc_key:
            stride_idx.append(0)

        for i in range(1, len(gc_event_label)):
            if gc_event_label[i] == gc_key and gc_event_label[i - 1] != gc_key:
                stride_idx.append(i)

        return stride_idx

    @staticmethod
    def normalize_per_gc(stride_idx, data):
        normalize_data = list()
        for i in range(len(stride_idx) - 1):
            normalize_data.append(data[stride_idx[i]:stride_idx[i + 1]])

        return normalize_data

    def interpolate_data(self, data):
        new_data = list()
        for stride in data:
            x = np.linspace(0, self.samples, num=stride.shape[0], endpoint=True)
            normalize_obj = interp1d(x, stride, kind='cubic')
            x_new = np.linspace(0, self.samples, num=self.samples, endpoint=True)
            new_stride = normalize_obj(x_new)
            new_data.append(new_stride)

        return new_data

    @staticmethod
    def label_strides_data(stride_data, stride_labels):
        data_and_labels = list()
        for stride, labels in zip(stride_data, stride_labels):
            activities = list(np.unique(labels))
            if len(activities) > 1:
                label = f"{labels[0].split('2')[0]}2{labels[-1].split('2')[0]}"
            else:
                label = activities[0]
            data_and_labels.append((stride, label))

        all_labels = np.unique([i[1] for i in data_and_labels])
        all_data_labelled = {label: list() for label in all_labels}

        for data, label in data_and_labels:
            all_data_labelled[label].append(data)

        for label in all_data_labelled.keys():
            all_data_labelled[label] = np.array(all_data_labelled[label])

        return all_data_labelled

    def save_files(self, data, subject, joint):
        path_lvl_1 = self.path
        if not os.path.exists(path_lvl_1):
            os.mkdir(path_lvl_1)
        path_lvl_2 = os.path.join(path_lvl_1, f"AB{str(subject).zfill(2)}")
        if not os.path.exists(path_lvl_2):
            os.mkdir(path_lvl_2)
        path_lvl_3 = os.path.join(path_lvl_2, f"{joint.split('_')[1]}Angles")
        if not os.path.exists(path_lvl_3):
            os.mkdir(path_lvl_3)

        for label, data_ in data.items():
            file_name = os.path.join(path_lvl_3, f"{label}.npy")
            np.save(file_name, data_)

    def segment_by_gc_key(self, data, subject):
        labels = data.pop('Labels')
        activity_label = labels[:, 0]
        gc_event_label = labels[:, 1]

        for joint, angle_values in data.items():
            sagittal_data = data[joint][:, 2]
            if joint.split('_')[0] == 'Right':
                gc_key = f"r{self.gc_key}"
            else:
                gc_key = f"l{self.gc_key}"

            stride_idx = self.get_stride_idx(gc_event_label=gc_event_label, gc_key=gc_key)
            normalize_labels = self.normalize_per_gc(stride_idx=stride_idx, data=activity_label)
            normalize_data = self.normalize_per_gc(stride_idx=stride_idx, data=sagittal_data)
            normalize_data = self.interpolate_data(normalize_data)

            all_data_labelled = self.label_strides_data(stride_data=normalize_data, stride_labels=normalize_labels)
            self.save_files(data=all_data_labelled, subject=subject, joint=joint)

    def run(self):
        for i in range(len(self.data_loader)):
            print(f"Subject {str(i + 1).zfill(2)}...")
            clean_data = self.remove_nones(data=self.data_loader[i])
            self.segment_by_gc_key(data=clean_data, subject=i + 1)
