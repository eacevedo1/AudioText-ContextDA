# Import libraries
import os
import sys

import git
import laion_clap

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)


def get_LAIONCLAP_model():
    """
    Returns the CLAP model with the best checkpoint

    Parameters:
    root_path (str): The root path of the project
    """
    # Load the LION-CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=True)

    # Load the best checkpoint
    ckpt_path = os.path.join(
        ROOT_DIR, "models", "LAION-CLAP", "630k-audioset-fusion-best.pt"
    )
    model.load_ckpt(ckpt=ckpt_path)

    return model
