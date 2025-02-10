# Import libraries
import os
import sys

import git
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_models as get_models


def get_text_anchors(label_map):
    """
    Get the text-anchors embeddings for the sound classification (class prototypes)

    Args:
        label_map: dictionary containing the class labels (sorted by the classID)

    Returns:
        text_features: tensor containing the text-anchors embeddings
    """

    # Create the prompt text
    prompt_text = ["This is a sound of " + value for key, value in label_map.items()]

    # Load the model
    get_model = getattr(get_models, "get_LAIONCLAP_model")
    model = get_model()

    # Get the text embeddings
    text_features = model.get_text_embedding(prompt_text)

    return torch.tensor(text_features)


def get_centroid_prototypes(label_map, train_embd, train_gt):
    """
    Get the centroid prototypes for the sound classification

    Args:
        label_map: dictionary containing the class labels (sorted by the classID)
        train_embd: tensor containing the training embeddings
        train_gt: tensor containing the training ground truth

    Returns:
        centroid_prototypes: tensor containing the centroid prototypes
    """

    # Create the zero tensor to store the centroid prototypes
    centroid_prototypes = torch.zeros(len(label_map), train_embd.shape[1])

    # Iterate over the classes
    for label_idx, label_name in label_map.items():

        # Get the indices of the current class
        idx = torch.where(train_gt == label_idx)[0]

        # Get the embeddings of the current class
        class_embd = train_embd[idx]

        # Get the centroid of the current class
        centroid_prototypes[label_idx] = torch.mean(class_embd, dim=0)

    return centroid_prototypes


def get_tgap_prototypes(label_map, train_embd, text_features, topn=35):
    """
    Get the text-guided audio prototypes for the sound classification

    Args:
        label_map: dictionary containing the class labels (sorted by the classID)
        train_embd: tensor containing the training embeddings
        text_features: tensor containing the text-anchors embeddings

    Returns:
        audio_prototypes: tensor containing the text-guided audio prototypes
    """

    # Create the zero tensor to store the text-guided audio prototypes
    audio_prototypes = torch.zeros(len(label_map), train_embd.shape[1])

    # Get the logits
    logits_audio_text = (train_embd @ text_features.t()).detach().cpu()

    # Iterate over the classes
    for label_idx, label_name in label_map.items():

        # Get the top-n with the highest confidence values in for the current class
        conf_values, idx = logits_audio_text[:, label_idx].topk(topn)

        # Get the embeddings of the top-n
        class_embd = train_embd[idx]

        # Get the text-guided audio prototype of the current class
        audio_prototypes[label_idx] = torch.mean(class_embd, dim=0)

    return audio_prototypes
