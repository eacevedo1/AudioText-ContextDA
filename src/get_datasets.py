# Import libraries
import os
from glob import glob

import git
import pandas as pd
from torch.utils.data import Dataset

# Root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir


# UrbanSound8k Dataset Class
class urbansound8k_Dataset(Dataset):
    """
    Dataset class for the UrbanSound8k dataset.
    """

    def __init__(self, **kwargs):
        super(urbansound8k_Dataset, self).__init__()

        # See if the folder path is provided
        if kwargs["folder_path"] is not None:
            self.dataset_folder_path = self.dataset_folder_path = os.path.join(
                ROOT_DIR, "data", "input", kwargs["folder_path"]
            )
        else:
            self.dataset_folder_path = os.path.join(
                ROOT_DIR, "data", "input", "urbansound8k", "audio"
            )

        # Get all audio files paths
        self.paths = glob(os.path.join(self.dataset_folder_path, "fold*/*.wav"))

        # Get the label map
        self.label_map = self.get_label_map()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path

    def get_fold_count(self):
        """
        Returns the number of folds in the dataset
        """
        return len(glob(self.dataset_folder_path + "fold*"))

    def get_label_map(self):
        """
        Returns the label map for the dataset
        """
        # Load the metadata
        labels_path = os.path.join(
            ROOT_DIR, "data", "input", "urbansound8k", "metadata", "UrbanSound8K.csv"
        )
        df = pd.read_csv(labels_path)

        # Change _ to space in the class column
        df["class_"] = df["class"].apply(lambda x: " ".join(x.split("_")))

        # Create a dictionary with the classID as the key and the class as the value
        label_map = dict(zip(list(df.classID), list(df.class_)))

        # Sort the dictionary by the classID
        label_map = dict(sorted(label_map.items()))

        return label_map


# TAU Urban Acoustic Scenes 2019 Dataset Class
class tau2019uas_Dataset(Dataset):
    """
    Dataset class for the TAU Urban Acoustic Scenes 2019 dataset.
    """

    def __init__(self, **kwargs):
        super(tau2019uas_Dataset, self).__init__()

        # See if the folder path is provided
        if kwargs["folder_path"] is None:
            self.dataset_folder_path = os.path.join(
                ROOT_DIR,
                "data",
                "input",
                "tau2019uas",
                "TAU-urban-acoustic-scenes-2019-development",
            )

        # Get all the paths of the audio files (both in subdirectories and directly in the audio folder)
        # This is for the case when the dataset is sorted or unsorted
        self.paths = glob(self.dataset_folder_path + "/audio/*/*.wav") + glob(
            self.dataset_folder_path + "/audio/*.wav"
        )

        # Get the label map
        self.label_map = self.get_label_map()

        # Get the fold dictionary
        self.fold_dict = self.get_fold_dict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path

    def get_label_map(self):
        """
        Returns the label map for the TAU Urban Acoustic Scenes 2019 dataset
        """

        # Load the metadata
        df = pd.read_csv(os.path.join(self.dataset_folder_path, "meta.csv"), sep="\t")

        # Get the unique labels
        labels = df["scene_label"].unique()

        # Create a dictionary with classID as key and label as value
        return dict(zip(range(len(labels)), labels))

    def get_fold_dict(self):
        """
        Returns the fold dictionary for the TAU Urban Acoustic Scenes 2019 dataset.
        This dictionary contains the fold number for each audio file.
        """

        # Load Train Metadata
        df_fold_train = pd.read_csv(
            os.path.join(
                self.dataset_folder_path, "evaluation_setup", "fold1_train.csv"
            ),
            sep="\t",
        )
        df_fold_train["fold"] = 1
        df_fold_train.drop(columns=["scene_label"], inplace=True)

        # Load Validation Metadata
        df_fold_val = pd.read_csv(
            os.path.join(
                self.dataset_folder_path, "evaluation_setup", "fold1_evaluate.csv"
            ),
            sep="\t",
        )
        df_fold_val["fold"] = 2
        df_fold_val.drop(columns=["scene_label"], inplace=True)

        # Load Test Metadata
        df_fold_test = pd.read_csv(
            os.path.join(
                self.dataset_folder_path, "evaluation_setup", "fold1_test.csv"
            ),
            sep="\t",
        )
        df_fold_test["fold"] = 3

        # Concatenate the dataframes to get the fold number for each audio file
        df = pd.concat([df_fold_train, df_fold_val, df_fold_test], axis=0)

        return df.set_index("filename")["fold"].to_dict()

    def get_fold(self, path):
        """
        Returns the fold number for the given audio filename.
        """
        try:
            return self.fold_dict["audio/" + path.split("/")[-1]]
        except:
            return 4


# Custom Dataset Class
class custom_Dataset(Dataset):
    """
    Dataset class for custom datasets.
    This dataset takes all the audio files in the dataset folder.
    """

    def __init__(self, **kwargs):
        """
        Parameters:
            dataset_folder_path (str): The path to the dataset folder
        """
        super(custom_Dataset, self).__init__()

        # Get the dataset folder path
        self.dataset_folder_path = self.dataset_folder_path = os.path.join(
            ROOT_DIR, "data", "input", kwargs["folder_path"]
        )

        # Get all the paths of the audio files
        self.paths = glob(self.dataset_folder_path + "/*.wav")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path
