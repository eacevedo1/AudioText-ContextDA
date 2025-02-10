###################################################################################################
# Description: This script is used to make sound classification using a pre-trained audio-text
#              model LION-CLAP.
#
# Usage: python scripts/sound_classification.py --embeddings_path <embeddings_path> --dataset <dataset> --mode <mode>
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

# Import libraries
import argparse
import os
import sys

import git
import numpy as np
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_datasets as get_datasets
from src.domain_adaptation_utils import (get_background_profile_audio,
                                         get_background_profile_text)
from src.get_embedding import load_embeddings
from src.metrics import accuracy_score
from src.sound_classification_utils import (get_centroid_prototypes,
                                            get_text_anchors,
                                            get_tgap_prototypes)


def main(args):

    # Set Temperature value
    TEMPERATURE = args.temperature

    # Define the embeddings dictionary object which contains the filenames, embeddings, ground truth and folds
    embd_dict_obj = load_embeddings(args.embeddings_path)

    # Load label map
    dataset = getattr(get_datasets, f"{args.dataset}_Dataset")
    train_set = dataset(folder_path=None)
    label_map = train_set.get_label_map()

    # Define the embeddings dictionary object whichi contains the filenames, embeddings, ground truth and folds of backgrounds
    if args.modality == "audio":
        bg_embd_dict_obj = load_embeddings(args.bg_embeddings_path)

    # Get the text-anchors embeddings
    text_features = get_text_anchors(label_map)

    # Get the background type for domain adaptation using text
    if args.modality == "text":
        # Extract the background types
        bg_types = [
            key.split("-")[-2].replace("_", " ") for key in embd_dict_obj.audios_key
        ]

        # Check if there are multiple background types
        if len(set(bg_types)) > 1:
            raise ValueError(f"Multiple background types found: {set(bg_types)}")

    # List to store the accuracies
    acc_list = []

    # Iterate over the folds of the dataset
    for fold in embd_dict_obj.folds:

        # Get train and test sets
        train_folds_list = [f for f in embd_dict_obj.folds if f != fold]
        train_key, train_embd, train_gt, train_fold = embd_dict_obj.get_set(
            train_folds_list
        )
        test_key, test_embd, test_gt, test_fold = embd_dict_obj.get_set([fold])

        # Mode zero-shot (zs) - Uses the text-anchors embeddings to classify the audio embeddings
        if args.mode == "zs":

            # Get the logits
            ss_profile = (test_embd @ text_features.t()).detach().cpu()

            # Get the background profile for domain adaptation
            if args.modality == "text":
                bg_profile = get_background_profile_text(bg_types[0], text_features)
            elif args.modality == "audio":
                bg_profile = get_background_profile_audio(
                    args.embeddings_path.split("_")[0],
                    bg_embd_dict_obj,
                    test_key,
                    test_fold,
                    text_features,
                )

        # Mode text-guided audio prototypes (tgap) - Uses the text-anchors embeddings to guide the audio prototypes
        elif args.mode == "tgap":

            # Get the text-guided audio prototypes
            audio_prototypes = get_tgap_prototypes(label_map, train_embd, text_features)

            # Get the logits
            ss_profile = (test_embd @ audio_prototypes.t()).detach().cpu()

            # Get the background profile for domain adaptation
            if args.modality == "text":
                bg_profile = get_background_profile_text(bg_types[0], audio_prototypes)
            elif args.modality == "audio":
                bg_profile = get_background_profile_audio(
                    args.embeddings_path.split("_")[0],
                    bg_embd_dict_obj,
                    test_key,
                    test_fold,
                    audio_prototypes,
                )

        # Mode supervised (sv) - Uses the centroids of each class to classify the audio embeddings
        elif args.mode == "sv":

            # Get the centroid prototypes
            centroid_prototypes = get_centroid_prototypes(
                label_map, train_embd, train_gt
            )

            # Get the logits
            ss_profile = (test_embd @ centroid_prototypes.t()).detach().cpu()

            # Get the background profile for domain adaptation
            if args.modality == "text":
                bg_profile = get_background_profile_text(
                    bg_types[0], centroid_prototypes
                )
            elif args.modality == "audio":
                bg_profile = get_background_profile_audio(
                    args.embeddings_path.split("_")[0],
                    bg_embd_dict_obj,
                    test_key,
                    test_fold,
                    centroid_prototypes,
                )

        # Apply domain adaptation if selected
        if args.modality is not None:
            ss_profile = ss_profile - (bg_profile * TEMPERATURE)

        # Apply softmax and get the predicted labels
        conf, idx = torch.softmax(ss_profile, dim=-1).topk(1, dim=-1)

        # Get the predicted labels
        y_pred = idx.squeeze().numpy()

        # Get the accuracy
        curr_acc = accuracy_score(test_gt, y_pred) * 100

        # Save the accuracy
        acc_list.append(curr_acc)

        # Print the accuracy for the current fold
        print(f" Fold={fold}, acc/mAP={curr_acc:.3f}%")

        # Calculate the mean accuracy
    mean_acc = np.mean(acc_list)
    print(
        f" Final score: Model=LAION-CLAP, train_type={args.mode}, acc/mAP={mean_acc:.3f}%"
    )

    # Save the results
    if args.save_results == "True":
        os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
        embedding_name = args.embeddings_path.split(".pt")[0]
        with open(
            os.path.join(
                ROOT_DIR,
                "results",
                f"{embedding_name}_results_{args.mode}_{args.modality}_{TEMPERATURE}.txt",
            ),
            "w",
        ) as f:
            f.write(f"Model=LAION-CLAP, train_type={args.mode}, acc/mAP={mean_acc:.3f}%")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Embeddings path where the embeddings dictionary is stored
    parser.add_argument(
        "--embeddings_path",
        type=str,
        help="Path to the embeddings file.",
    )

    # Dataset name (urbansound8k, ...)
    parser.add_argument(
        "--dataset",
        type=str,
        help="E.g. urbansound8k, ...",
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

    # Embeddings path where the background embeddings dictionary is stored
    # Only needed for Audio domain adaptation
    parser.add_argument(
        "--bg_embeddings_path",
        type=str,
        help="Path to the embeddings file.",
    )

    # Define train type (zero-shot(zs), text-guided audio prototypes(tgap) or supervised(sv))
    parser.add_argument("--mode", type=str, default="zs", help="zs, tgap, sv")

    # Save results
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="'True' to save the results as a txt file.",
    )

    args = parser.parse_args()
    main(args)
