###################################################################################################
# Description: Code to generate soundscapes for the UrbanSound8k dataset as foreground and using
#              the TAU Urban Acoustic Scenes 2019 dataset as background.
#
# Usage: python soundscape_augmentations.py --folds 1,2,3,4,5,6,7,8,9,10 \
#                                           --parameters ref_db=-36 seed=123 n_soundscapes=1 duration=10.0 event_time=(truncnorm,3.0,1.5,0.0,6.0) snr_dist=(uniform,6,10) bg=all
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
import numpy as np
import scaper

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)


def convert_to_type(value):
    """
    Convert a string to the correct type.

    Parameters:
        value (str): String with the value.

    Returns:
        value: Value converted to the correct type.
    """

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_params(args):
    """
    Get the parameters as tuples or single values.

    Parameters:
        args (Namespace): Namespace with the arguments.

    Returns:
        params (dict): Dictionary with the parameters of the transformation.
    """

    # Convert parameters to a dictionary
    params = dict(param.split("=") for param in args.parameters[0].split(" "))

    # Convert values to the correct type
    for key, value in params.items():
        # Check if the value is a tuple
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1].split(",")
            value = tuple(convert_to_type(v) for v in value)
        else:
            value = convert_to_type(value)

        params[key] = value

    return params


def save_parameters(output_folder, params):
    """
    Save the parameters of the transformation in a text file.

    Parameters:
        output_folder (str): Path to the output folder.
        params (dict): Dictionary with the parameters of the transformation.
    """
    with open(f"{output_folder}/parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    return


def main(args):

    # Output folder
    outfolder = os.path.join(
        ROOT_DIR,
        "data",
        "input",
        "urbansound8k-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    os.makedirs(outfolder, exist_ok=True)

    # Scaper settings
    fg_folder = os.path.join(ROOT_DIR, "data", "input", "urbansound8k", "audio")
    bg_folder = os.path.join(
        ROOT_DIR,
        "data",
        "input",
        "tau2019uas",
        "TAU-urban-acoustic-scenes-2019-development",
        "audio",
    )

    # Check if there are folders in the background folder (if not, run the script to order the dataset)
    folders = [
        f for f in os.listdir(bg_folder) if os.path.isdir(os.path.join(bg_folder, f))
    ]

    if len(folders) == 0:
        print("Ordering the TAU Urban Acoustic Scenes 2019 dataset")
        # Run the script to order the TAU Urban Acoustic Scenes 2019 dataset
        os.system(f"python3 {ROOT_DIR}/src/order_tau_dataset.py")

    # Parse parameters
    params = get_params(args)

    # Set default values for the parameters
    defaults = {
        "ref_db": -36,
        "seed": 123,
        "n_soundscapes": 1,
        "duration": 10.0,
        "event_time": ("truncnorm", 3.0, 1.5, 0.0, 6.0),
        "snr_dist": ("uniform", 6, 10),  # ('const', 6)
        "bg": "all",  # 'all' or 'park' or 'airport'
    }

    # Update the defaults with the parameters
    for key, default_value in defaults.items():
        params.setdefault(key, default_value)

    # Save a text file on output folder with transformation parameters
    save_parameters(outfolder, params)

    # Folds to use for generating soundscapes
    folds = args.folds.split(",")
    folds = [int(f) for f in folds]

    # Iterate over the folds
    for fold in folds:

        # List of all the audio files in the fold (finishing in .wav)
        audio_files = [
            f
            for f in os.listdir(os.path.join(fg_folder, f"fold{fold}"))
            if f.endswith(".wav")
        ]

        # On the output folder, create a folder for each fold and inside it, create a txt and jams folder
        outfolder_fold = os.path.join(outfolder, f"fold{fold}")

        # Create the output folder
        os.makedirs(outfolder, exist_ok=True)
        os.makedirs(os.path.join(outfolder_fold, "txt"), exist_ok=True)
        os.makedirs(os.path.join(outfolder_fold, "jams"), exist_ok=True)

        # Create a Scaper object
        sc = scaper.Scaper(
            duration=params["duration"],
            fg_path=fg_folder,
            bg_path=bg_folder,
            random_state=params["seed"],
        )

        # Reference dB
        sc.ref_db = params["ref_db"]

        # Iterate over the audio files
        for file in audio_files:

            # Sample n_soundscapes folder to use as background from bg_folder
            if params["bg"] == "all":
                bg_folder_name = [
                    f
                    for f in os.listdir(bg_folder)
                    if os.path.isdir(os.path.join(bg_folder, f))
                ]
                bg_folder_sample = np.random.choice(
                    bg_folder_name, params["n_soundscapes"]
                )
            else:
                bg_folder_sample = np.array([params["bg"]] * params["n_soundscapes"])

            # File path
            file_path = os.path.join(fg_folder, f"fold{fold}", file)

            for i, bg in enumerate(bg_folder_sample):

                # reset the event specifications for foreground and background at the
                # beginning of each loop to clear all previously added events
                sc.reset_fg_event_spec()
                sc.reset_bg_event_spec()

                # Add background
                sc.add_background(
                    label=("choose", [bg]),
                    source_file=("choose", []),
                    source_time=("const", 0),
                )

                # Add foreground event
                sc.add_event(
                    label=("choose", [f"fold{fold}"]),
                    source_file=("choose", [file_path]),
                    source_time=("const", 0),
                    event_time=params["event_time"],
                    event_duration=("const", 4),
                    snr=params["snr_dist"],
                    pitch_shift=None,
                    time_stretch=None,
                )

                # Generate
                file_name = file.split(".")[0] + "-" + bg + f"-{i}"
                audiofile = os.path.join(outfolder_fold, f"{file_name}.wav")
                jamsfile = os.path.join(outfolder_fold, "jams", f"{file_name}.jams")
                txtfile = os.path.join(outfolder_fold, "txt", f"{file_name}.txt")

                sc.generate(
                    audiofile,
                    jamsfile,
                    allow_repeated_label=True,
                    allow_repeated_source=False,
                    reverb=0,
                    disable_sox_warnings=True,
                    no_audio=False,
                    txt_path=txtfile,
                )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate soundscapes for UrbanSound8k dataset."
    )

    # Define folds to use for generating soundscapes
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="List of Folds to be used",
    )

    # Define parameters for the transformation
    parser.add_argument(
        "--parameters",
        type=str,
        nargs="*",
        help="Function parameters in the format key1=(value1,value2) key2=value2 ...",
    )

    args = parser.parse_args()
    main(args)
