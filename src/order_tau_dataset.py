# This script is used to order the TAU Urban Acoustic Scenes 2019 dataset for the use of Scaper.

# Imports
import argparse
import os
import shutil
import sys

import git

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)


def main(args):
    # Create a set of classes
    folder_path = os.path.join(ROOT_DIR, "data", "input", args.folder_path)
    classes = set()

    for file in os.listdir(folder_path):
        class_name = file.split("-")[0]
        classes.add(class_name)

    # Create a folder for each class
    for class_name in classes:
        if not os.path.exists(class_name):
            os.makedirs(os.path.join(folder_path, class_name))

    # Move the files to the corresponding folder (only if the file is a wav file)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            class_name = file.split("-")[0]
            shutil.move(
                os.path.join(folder_path, file),
                os.path.join(folder_path, class_name, file),
            )

    print("Files ordered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        type=str,
        default="tau2019uas/TAU-urban-acoustic-scenes-2019-development/audio/",
        help="Path to the folder with the audio files",
    )
    args = parser.parse_args()

    main(args)
