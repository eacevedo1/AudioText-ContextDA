# Import Libraries
import os
import sys

import git
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_models as get_models


# Function to get the embeddings for the UrbanSound8k dataset
def get_urbansound8k_embeddings(train_set, feat_data, args):
    """
    Returns the embeddings for the UrbanSound8k dataset
    """
    # Load the dataset
    train_dataloader = DataLoader(
        train_set, batch_size=64, shuffle=False, num_workers=args.num_workers
    )

    # Load the model
    get_model = getattr(get_models, f"get_LAIONCLAP_model")
    model = get_model()

    # Iterate over the dataset
    for paths in tqdm(train_dataloader):
        audio_embd = None

        # Get the embeddings of a batch of audio files
        audio_embd = model.get_audio_embedding_from_filelist(paths)

        # Store the embeddings and metadata
        for idx, embd in enumerate(audio_embd):
            # Get the file name, label and fold of each audio file of the batch
            path = paths[idx]
            file_name, label, fold = None, None, None

            file_name = path.split("/")[-1]
            label = int(file_name.split(".")[0].split("-")[1])
            fold = int(path.split("/")[-2].split("fold")[1])

            # Store the label, embeddings and fold
            feat_data[file_name] = {}
            feat_data[file_name]["class_gt"] = label
            feat_data[file_name]["embd"] = embd
            feat_data[file_name]["fold"] = fold

    return feat_data


def get_tau2019uas_embeddings(train_set, feat_data, args):
    """
    Returns the embeddings for the TAU Urban Acoustic Scenes 2019 dataset
    """
    # Load the dataset
    train_dataloader = DataLoader(
        train_set, batch_size=64, shuffle=False, num_workers=args.num_workers
    )

    # Load the model
    get_model = getattr(get_models, f"get_LAIONCLAP_model")
    model = get_model()

    # Change the key and value of the label map
    label_map = {v: k for k, v in train_set.label_map.items()}

    # Iterate over the dataset
    for paths in tqdm(train_dataloader):
        audio_embd = None

        # Get the embeddings of a batch of audio files
        audio_embd = model.get_audio_embedding_from_filelist(paths)

        # Store the embeddings and metadata
        for idx, embd in enumerate(audio_embd):
            path = paths[idx]

            # Get the file name, label and fold
            file_name, label, fold = None, None, None

            file_name = path.split("/")[-1]
            label = label_map[path.split("/")[-2]]
            fold = train_set.get_fold(file_name)

            # Store the label, embeddings and fold
            feat_data[file_name] = {}
            feat_data[file_name]["class_gt"] = label
            feat_data[file_name]["embd"] = embd
            feat_data[file_name]["fold"] = fold

    return feat_data


# Class to load the embeddings for any dataset
class load_embeddings:
    """
    Class to load the embeddings
    """

    def __init__(self, embeddings_dict_filename):

        self.embeddings_dict_filename = embeddings_dict_filename

        # Load the embeddings dictionary
        self.embeddings_dict = torch.load(
            os.path.join(ROOT_DIR, "data", "embeddings", self.embeddings_dict_filename)
        )

        # Load the keys, embeddings, ground truth labels and folds
        self.audios_key, self.audios_embd, self.audios_gt, self.audios_fold = (
            self.load_embeddings(self.embeddings_dict)
        )

        # Get the folds numbers
        self.folds = set(np.unique(self.audios_fold))

        return

    def load_embeddings(self, embeddings_dict):
        """
        Load the embeddings, ground truth labels and folds from a dictionary

        Args:
            embeddings_dict: dictionary containing the embeddings, ground truth labels and folds

        Returns:
            audios_embd: tensor containing the embeddings
            audios_gt: tensor containing the ground truth labels
            audios_fold: tensor containing the folds
        """

        # Initialize the lists
        audios_key = []
        audios_embd = []
        audios_gt = []
        audios_fold = []

        # Iterate over the dictionary
        for key in embeddings_dict.keys():
            audios_key.append(key)
            audios_embd.append(torch.tensor(embeddings_dict[key]["embd"]))
            audios_gt.append(embeddings_dict[key]["class_gt"])
            audios_fold.append(embeddings_dict[key]["fold"])

        # Convert the lists to tensors
        audios_key = np.array(audios_key)
        audios_embd = torch.stack(audios_embd)
        audios_gt = torch.tensor(audios_gt)
        audios_fold = torch.tensor(audios_fold)

        return audios_key, audios_embd, audios_gt, audios_fold

    def get_set(self, folds_list):
        """
        Returns the embeddings, ground truth labels and folds for a given list of folds

        Args:
            folds_list: list of folds

        Returns:
            audios_embd: tensor containing the embeddings
            audios_gt: tensor containing the ground truth labels
            audios_fold: tensor containing the folds
        """

        # Get the indices of the embeddings for the given folds
        idxs = [i for i, x in enumerate(self.audios_fold) if x in folds_list]

        # Get the embeddings, ground truth labels and folds for the given folds
        audios_key = self.audios_key[idxs]
        audios_embd = self.audios_embd[idxs]
        audios_gt = self.audios_gt[idxs]
        audios_fold = self.audios_fold[idxs]

        return audios_key, audios_embd, audios_gt, audios_fold


# Function to get the embeddings for the Custom dataset
def get_custom_embeddings(train_set, feat_data, args):
    """
    Returns the embeddings for the Custom dataset
    """
    # Load the dataset
    train_dataloader = DataLoader(
        train_set, batch_size=64, shuffle=False, num_workers=args.num_workers
    )

    # Load the model
    get_model = getattr(get_models, f"get_LAIONCLAP_model")
    model = get_model()

    # Iterate over the dataset
    for paths in tqdm(train_dataloader):
        audio_embd = None

        # Get the embeddings of a batch of audio files
        audio_embd = model.get_audio_embedding_from_filelist(paths)

        # Store the embeddings and metadata
        for idx, embd in enumerate(audio_embd):
            # Get the file name, label and fold of each audio file of the batch
            path = paths[idx]
            file_name = None
            file_name = path.split("/")[-1]

            # Store the label, embeddings and fold
            feat_data[file_name] = {}
            feat_data[file_name]["embd"] = embd

    return feat_data
