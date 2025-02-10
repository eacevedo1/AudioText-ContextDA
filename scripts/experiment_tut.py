###################################################################################################
# Description: This script classifies audio files using a domain adaptation technique. It calculates
#              embeddings for audio files and text-anchors, applies domain adaptation based on the
#              specified modality (text or audio), and computes the classification accuracy.
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

# Import libraries
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import git
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_datasets as get_datasets
from src.domain_adaptation_utils import (
    get_background_profile_audio_inference, get_background_profile_text)
from src.get_embedding import get_custom_embeddings
from src.inference_utils import (get_label_map_inference,
                                 parse_embeddings_inference,
                                 parse_embeddings_inference_tut,
                                 save_results_inference)
from src.sound_classification_utils import get_text_anchors


def main(args):

    # Set Temperature value
    TEMPERATURE = args.temperature

    # Load label map from the class labels file
    label_map = get_label_map_inference(args.class_labels)

    # Get the text-anchors embeddings
    text_features = get_text_anchors(label_map)

    # Load the dataset for the test audios files
    dataset = getattr(get_datasets, f"custom_Dataset")
    test_set = dataset(folder_path=args.audio_folder_path)

    # Calculate the embeddings for the audio files to be classified
    test_dict = {}
    test_dict = get_custom_embeddings(test_set, test_dict, args)
    test_keys, test_embd, audio_labels = parse_embeddings_inference_tut(test_dict, args)

    # Get the logits
    ss_profile = (test_embd @ text_features.t()).detach().cpu()

    # Get the background profile for domain adaptation
    if args.modality == "text":
        bg_profile = get_background_profile_text(args.bg_type, text_features)
    elif args.modality == "audio":
        # Get the background dataset
        background_set = dataset(folder_path=args.bg_folder_path)

        # Calculate the embeddings for the background audio files
        bg_dict = {}
        bg_dict = get_custom_embeddings(background_set, bg_dict, args)
        bg_keys, bg_embd = parse_embeddings_inference(bg_dict)

        # Get the background profile
        bg_profile = get_background_profile_audio_inference(bg_embd, text_features)

    # Apply domain adaptation if selected
    if args.modality is not None:
        ss_profile = ss_profile - (bg_profile * TEMPERATURE)

    # Apply softmax and get the predicted labels
    conf = torch.softmax(ss_profile, dim=-1)

    # Get the dictionary to get the index of the true label
    label_to_index = {value: key for key, value in label_map.items()}

    # Get the accuracy
    acc = []
    for conf_i, labels_i in zip(conf, audio_labels):

        # Convert the list to a tensor of indices
        indices = [label_to_index[label] for label in labels_i]
        indices_tensor = torch.tensor(indices)

        # Get top k predictions where k matches the length of indices_tensor
        k = len(indices_tensor)
        top_k_conf, top_k_idx = torch.topk(conf_i, k=k)

        # Calculate the accuracy
        acc_i = 0
        for idx_gt in indices_tensor:
            if idx_gt in top_k_idx:
                acc_i += 1
        acc.append(acc_i / k)

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path file text with the labels names
    parser.add_argument(
        "--class_labels",
        type=str,
        default=None,
        help="Path to the file containing the class labels.",
    )

    # All audio folders
    parser.add_argument(
        "--all_audio_folders",
        type=str,
        default=None,
        help="All audio folders.",
    )

    # Path to the folder containing the audio to be classified
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        default=None,
        help="Path to the folder containing the audio to be classified.",
    )

    # Domain Adaptation Modality
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Modality to be used for domain adaptation. E.g. 'text' or 'audio'. None for no domain adaptation.",
    )

    # Temperature for domain adaptation
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature to be used for domain adaptation.",
    )

    # Background type for domain adaptation
    # Only needed for TEXT domain adaptation
    parser.add_argument(
        "--bg_type",
        type=str,
        default=None,
        help="Determines the type of background. E.g. 'park', 'airport', 'street traffic', ...",
    )

    # Path to folder of background audios
    # Only needed for AUDIO domain adaptation
    parser.add_argument(
        "--bg_folder_path",
        type=str,
        default=None,
        help="Path to the background audio folder.",
    )

    # Number of workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for extracting the embeddings.",
    )

    args = parser.parse_args()

    # Get the folders in the audio_folder_path
    args.all_audio_folders = os.path.join(
        ROOT_DIR, "data", "input", args.all_audio_folders
    )

    audio_folders = [
        d.name for d in Path(args.all_audio_folders).iterdir() if d.is_dir()
    ]

    mean_accs = []
    for audio in tqdm(audio_folders):
        args.audio_folder_path = os.path.join(args.all_audio_folders, audio, "labeled")
        args.bg_folder_path = os.path.join(args.all_audio_folders, audio, "unlabeled")

        # If args.bg_folder_path has no files in it, skip the audio folder
        if len(os.listdir(args.bg_folder_path)) == 0:
            print(f"Skipping {audio} because it has no files in {args.bg_folder_path}")
            continue

        acc = main(args)
        mean_accs.extend(acc)

    # Save the mean accuracy
    np.save(
        f"mean_acc_{args.modality}_{args.temperature}_{args.bg_type}.npy",
        np.array(mean_accs),
    )

    # Print modality - temperature - mean accuracy
    print(f"{args.modality} - {args.temperature} - {np.mean(mean_accs)}")
