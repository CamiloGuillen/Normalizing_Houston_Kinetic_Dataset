import os
import numpy as np

from clean_dataset.outlier_detector import OutlierDetector


class DataCleaner:
    def __init__(self, base_path, data_loader, outlier_method='MCD'):
        self.base_path = base_path
        self.data_loader = data_loader
        self.outlier_detector = OutlierDetector(method=outlier_method)

    def get_all_data(self, subjects):
        print("Collecting all data...")
        all_data, all_labels = list(), list()

        for subject in subjects:
            joints = os.listdir(os.path.join(self.base_path, subject))
            for joint in joints:
                activities = os.listdir(os.path.join(self.base_path, subject, joint))
                for activity in activities:
                    data, labels = self.data_loader.get_data(subject=subject, joint=joint,
                                                             activity=str(activity).split('.')[0])
                    all_data.append(data)
                    all_labels.append(labels)

        all_data, all_labels = np.vstack(all_data), np.vstack(all_labels)

        return all_data, all_labels

    def detect_outliers(self, data, labels):
        print("Detecting outliers...")
        unique_joints = np.unique(labels[:, 1])
        unique_activities = np.unique(labels[:, 2])

        outliers_idx = list()
        for joint in unique_joints:
            for activity in unique_activities:
                data_ts = [data for i, data in enumerate(data) if joint == labels[i, 1] and
                           activity == labels[i, 2]]
                data_idx = [i for i, _ in enumerate(data) if
                            joint == labels[i, 1] and activity == labels[i, 2]]

                if len(data_ts) > 5:
                    outlier_detected = self.outlier_detector.detect_outliers(x=np.array(data_ts))
                    outliers_idx.append([data_idx[i] for i in outlier_detected])

        outliers_idx = np.concatenate(outliers_idx)

        return outliers_idx

    @staticmethod
    def structure_data(data, labels):
        print("Structuring data...")
        subjects, joints, activities = np.unique(labels[:, 0]), np.unique(labels[:, 1]), np.unique(labels[:, 2])

        structured_data = {subject: {joint: {activity: list() for activity in activities} for joint in joints}
                           for subject in subjects}

        for data_, label in zip(data, labels):
            structured_data[label[0]][label[1]][label[2]].append(data_.reshape(1, -1))

        structured_data = {subject: {joint: {activity: np.concatenate(structured_data[subject][joint][activity])
                                             for activity in activities
                                             if len(structured_data[subject][joint][activity]) > 0} for joint in joints}
                           for subject in subjects}

        return structured_data

    def run(self, subjects):
        all_data, all_labels = self.get_all_data(subjects=subjects)
        outliers_idx = self.detect_outliers(data=all_data, labels=all_labels)
        new_data = np.delete(all_data, outliers_idx, axis=0)
        new_labels = np.delete(all_labels, outliers_idx, axis=0)
        structured_data = self.structure_data(data=new_data, labels=new_labels)

        return structured_data

    @staticmethod
    def save(data, base_folder):
        print("Saving data...")
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)

        for subject, joint_dict in data.items():
            path_lvl_1 = os.path.join(base_folder, subject)
            if not os.path.exists(path_lvl_1):
                os.mkdir(path_lvl_1)

            for joint, activity_dict in joint_dict.items():
                path_lvl_2 = os.path.join(path_lvl_1, joint)
                if not os.path.exists(path_lvl_2):
                    os.mkdir(path_lvl_2)

                for activity, activity_data in activity_dict.items():
                    path_lvl_3 = os.path.join(path_lvl_2, f"{activity}.npy")
                    print(activity_data.shape)
                    np.save(path_lvl_3, arr=activity_data)

        return True

    def run_and_save(self, subjects, base_folder="clean_dataset"):
        structured_data = self.run(subjects=subjects)
        self.save(data=structured_data, base_folder=base_folder)

        return True
