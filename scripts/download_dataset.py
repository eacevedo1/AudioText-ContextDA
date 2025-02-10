###################################################################################################
# Description: Code to download the dataset from the soundata library.
#
# Info of all the datasets available
# https://soundata.readthedocs.io/en/latest/source/quick_reference.html
#
# Usage: python download_dataset.py --dataset_name urbansound8k
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

# Import libraries
import argparse
import os
import sys

import git
import soundata

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)


def main(args):

    # Download the dataset
    dataset = soundata.initialize(
        args.dataset_name,
        data_home=os.path.join(ROOT_DIR, "data", "input", args.dataset_name),
    )
    dataset.download(cleanup=True, force_overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments to name of the dataset to download
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="esc50, urbansound8k, fsd50k, tau2019uas",
    )
    args = parser.parse_args()

    print("Downloading dataset: ", args.dataset_name)

    main(args)
