# Import libraries
import os
import sys

import git
import jams
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_models as get_models


def get_background_profile_text(bg_type, prototypes):
    """
    Get the background profile embeddings for domain adaptation

    Args:
        bg_type: string containing the background type
        prototypes: tensor containing the prototypes of the classes

    Returns:
        bg_embd: tensor containing the background profile embeddings
    """

    # Load the model
    get_model = getattr(get_models, "get_LAIONCLAP_model")
    model = get_model()

    # Create the prompt text
    prompts = [
        "This is a sound of " + bg_type,
        bg_type.capitalize() + " sounds in the background",
        "This is a sound of " + bg_type + " in the background",
    ]

    # Get the text embeddings
    bg_embd = model.get_text_embedding(prompts)

    # Get the background profile embeddings
    background_profiles = (torch.tensor(bg_embd) @ prototypes.t()).detach().cpu()

    # Take the mean of the background profile embeddings
    background_profile = background_profiles.mean(dim=0)

    return background_profile


def get_background_profile_audio(
    dataset_folder, bg_embd_dict_obj, test_key, test_fold, prototypes
):

    # Create an empty tensor to store the background profile embeddings
    backgrounds_embddings = torch.zeros(test_key.shape[0], 512)

    # Iterate over test files
    for i, (key_file, fold_file) in enumerate(zip(test_key, test_fold.tolist())):
        # Get the background sound file
        jam_folder = os.path.join(
            ROOT_DIR,
            "data",
            "input",
            dataset_folder,
            "fold" + str(int(fold_file)),
            "jams",
        )
        jam = jams.load(os.path.join(jam_folder, key_file.split(".wav")[0] + ".jams"))

        # Get the annotation
        ann = jam.annotations.search(namespace="scaper")[0]

        # Get the background and foreground files
        for obs in ann.data:
            if obs.value["role"] == "background":
                bg_file = obs.value["source_file"].split("/")[-1]

            elif obs.value["role"] == "foreground":
                fg_file = obs.value["source_file"].split("/")[-1]

        # Get the embedding of the background sound
        backgrounds_embddings[i] = torch.tensor(
            bg_embd_dict_obj.embeddings_dict[bg_file]["embd"]
        )

    # Get the background profile embeddings
    background_profile = (backgrounds_embddings @ prototypes.t()).detach().cpu()

    return background_profile


def get_background_profile_audio_inference(bg_embd, prototypes):

    # Get the background profile embeddings
    background_profiles = (bg_embd @ prototypes.t()).detach().cpu()

    # Take the mean of the background profile embeddings
    background_profile = background_profiles.mean(dim=0)

    return background_profile
