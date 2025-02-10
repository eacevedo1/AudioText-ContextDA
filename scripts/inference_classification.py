###################################################################################################
# Description: This script allows to use the domain adaptation technique to classify audio files
#              using the embeddings of the audio files and the text-anchors.
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

# Import libraries
import argparse
import sys
from datetime import datetime

import git
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_datasets as get_datasets
from src.domain_adaptation_utils import (
    get_background_profile_audio_inference, get_background_profile_text)
from src.get_embedding import get_custom_embeddings
from src.inference_utils import (get_label_map_inference,
                                 parse_embeddings_inference,
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
    test_keys, test_embd = parse_embeddings_inference(test_dict)

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
    conf, idx = torch.softmax(ss_profile, dim=-1).topk(1, dim=-1)

    # Save the results
    now = datetime.now().strftime("%H%M%S")
    save_results_inference(
        test_keys, idx, conf, label_map, filename=f"results_{now}.csv"
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path file text with the labels names
    parser.add_argument(
        "--class_labels",
        type=str,
        default=None,
        help="Path to the file containing the class labels.",
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
    main(args)
