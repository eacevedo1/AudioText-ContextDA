import os
import sys

import git
import numpy as np
import pandas as pd
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)


def get_label_map_inference(file_name):
    """
    Function to read the file containing the class labels and return a dictionary with the labels.

    Args:
        file_name: str, file name containing the class labels.

    Returns:
        label_map: dict, dictionary with the class labels.
    """

    # Get the path to the file
    file_path = os.path.join(ROOT_DIR, "data", "input_text", file_name)

    # Now, reading the file and converting it into a dictionary
    with open(file_path, "r") as file:
        lines_from_file = file.readlines()

    # Converting the lines into a dictionary
    label_map = {index: line.strip() for index, line in enumerate(lines_from_file)}

    return label_map


def parse_embeddings_inference(embeddings_dict):

    # Initialize the lists
    audios_key = []
    audios_embd = []

    # Iterate over the dictionary
    for key in embeddings_dict.keys():
        audios_key.append(key)
        audios_embd.append(torch.tensor(embeddings_dict[key]["embd"]))

    # Convert the lists to tensors
    audios_key = np.array(audios_key)
    audios_embd = torch.stack(audios_embd)

    return audios_key, audios_embd

def parse_embeddings_inference_tut(embeddings_dict, args):

    file_path = os.path.join(
        ROOT_DIR, "data", "input", args.audio_folder_path
    )

    # Initialize the lists
    audios_key = []
    audios_embd = []
    audio_labels = []

    # Iterate over the dictionary
    for key in embeddings_dict.keys():
        audios_key.append(key)
        audios_embd.append(torch.tensor(embeddings_dict[key]["embd"]))

        chunk_labels = []
        with open(os.path.join(file_path, key.split(".")[0] + ".wav_labels.txt"), "r") as file:
            for line in file:
                chunk_label = line.strip()
                chunk_labels.append(chunk_label)

        audio_labels.append(chunk_labels)

    # Convert the lists to tensors
    audios_key = np.array(audios_key)
    audios_embd = torch.stack(audios_embd)

    return audios_key, audios_embd, audio_labels

def save_results_inference(test_keys, idx, conf, label_map, filename="results.csv"):

    # Save the results as csv filename, label, confidence
    results = []
    for i, key in enumerate(test_keys):
        results.append([key, label_map[idx[i].item()], conf[i].item()])
    results = pd.DataFrame(results, columns=["filename", "label", "confidence"])

    # Make results folder if it does not exist in the root directory
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

    # Save the results
    results.to_csv(
        os.path.join(ROOT_DIR, "results", filename),
        index=False,
    )
    return
