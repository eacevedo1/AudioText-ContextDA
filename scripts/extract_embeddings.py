###################################################################################################
# Description: Code to extract embeddings from the audio and text data.
# Usage: python extract_embddings.py --dataset urbansound8k --path urbansound8k-20240705184401
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

# Import libraries
import argparse
import datetime
import os
import sys

import git
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_datasets as get_datasets
import src.get_embedding as get_embedding


def main(args):

    # Load the dataset
    dataset = getattr(get_datasets, f"{args.dataset}_Dataset")
    train_set = dataset(folder_path=args.path)

    # Dictionary to store the embeddings and metadata
    feat_data = {}

    # Get the embeddings
    get_embd = getattr(get_embedding, f"get_{args.dataset.lower()}_embeddings")
    feat_data = get_embd(train_set, feat_data, args)

    # Save the embeddings
    now = datetime.datetime.now().strftime("%H%M")
    save_path = os.path.join(
        ROOT_DIR,
        "data",
        "embeddings",
        f"{args.dataset}_{now}.pt" if args.path is None else f"{args.path}_{now}.pt",
    )
    torch.save(feat_data, save_path)
    print(f"Embeddings saved at: {save_path}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to extract embeddings.")

    # Dataset name
    parser.add_argument(
        "--dataset",
        type=str,
        help="E.g. urbansound8k, ...",
    )

    # Folder path
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path of the augmented dataset, if isn't augmented set to None. E.g. urbansound8k-20240705184401",
    )

    # Number of workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for extracting the embeddings.",
    )

    args = parser.parse_args()

    # Set the device
    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"

    main(args)
