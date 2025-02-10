###################################################################################################
# Description: This script processes audio files and their corresponding annotation files by
#              splitting the audio into chunks of fixed duration. It categorizes the chunks into
#              labeled and unlabeled based on the presence of labels in the annotations. The
#              processed chunks are saved into separate folders for labeled and unlabeled data.
#
# Updated by: Emiliano Acevedo
# Updated on: 09/2024
###################################################################################################

import glob
import os

import librosa
import pandas as pd
import soundfile as sf


def process_audio(
    ann_folder, audio_folder, output_folder, dataset_type, chunk_duration=10
):
    """
    Process audio files and annotations, splitting audio into chunks of fixed duration
    and categorizing them based on whether they have labels.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each annotation file
    for ann_file in glob.glob(os.path.join(ann_folder, "**", "*.ann"), recursive=True):
        # Read annotation file based on dataset type
        if dataset_type == "tut2017se":
            ann = pd.read_csv(
                ann_file,
                sep="\t",
                header=None,
                names=["filename", "location", "start", "end", "label", "type", "id"],
            )
            audio_name = ann_file.split("/")[-1].replace(".ann", ".wav")
        elif dataset_type == "tut2017se-evaluation":
            ann = pd.read_csv(
                ann_file,
                sep="\t",
                header=None,
                names=[
                    "start",
                    "end",
                    "label",
                ],
            )
            audio_name = ann_file.split("/")[-1].replace(".ann", ".wav")
        else:  # tut2016
            ann = pd.read_csv(
                ann_file, sep="\t", header=None, names=["start", "end", "label"]
            )
            # Remove the (object) from the label
            ann["label"] = ann["label"].str.replace("(object) ", "")
            # audio_name = ann_file.split('/')[-1].replace('_full.ann', '.wav')
            audio_name = ann_file.split("/")[-1].replace(".ann", ".wav")

        audio_path = os.path.join(audio_folder, audio_name)

        # Skip if audio file doesn't exist
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_name} not found, skipping.")
            continue

        # Create specific output folder for the audio file
        audio_output_folder = os.path.join(
            output_folder, os.path.splitext(audio_name)[0]
        )
        os.makedirs(audio_output_folder, exist_ok=True)
        labeled_folder = os.path.join(audio_output_folder, "labeled")
        unlabeled_folder = os.path.join(audio_output_folder, "unlabeled")
        os.makedirs(labeled_folder, exist_ok=True)
        os.makedirs(unlabeled_folder, exist_ok=True)

        # Load audio using librosa
        audio, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
        duration = librosa.get_duration(y=audio, sr=sr)

        # Chop into chunks
        chunk_start = 0

        while chunk_start < duration:
            chunk_end = min(chunk_start + chunk_duration, duration)

            # Extract chunk samples
            start_sample = int(chunk_start * sr)
            end_sample = int(chunk_end * sr)
            chunk_audio = audio[start_sample:end_sample]

            # Determine labels for this chunk
            chunk_labels = ann[(ann["start"] < chunk_end) & (ann["end"] > chunk_start)][
                "label"
            ].unique()

            # Decide folder based on labels
            if len(chunk_labels) > 0:
                chunk_folder = labeled_folder
            else:
                chunk_folder = unlabeled_folder

            # Save chunk audio file
            chunk_name = f"chunk_{int(chunk_start)}-{int(chunk_end)}.wav"
            chunk_path = os.path.join(chunk_folder, chunk_name)
            sf.write(chunk_path, chunk_audio, sr)

            # Save labels if available
            if len(chunk_labels) > 0:
                labels_path = os.path.join(chunk_folder, f"{chunk_name}_labels.txt")
                with open(labels_path, "w") as f:
                    for label in chunk_labels:
                        f.write(f"{label}\n")

            # Update chunk times
            chunk_start += chunk_duration

    print("Processing complete!")


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process audio and annotation files into labeled and unlabeled chunks."
    )
    parser.add_argument(
        "--ann_folder", type=str, required=True, help="Path to the annotations folder."
    )
    parser.add_argument(
        "--audio_folder",
        type=str,
        required=True,
        help="Path to the audio files folder.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save the output chunks.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["tut2016", "tut2017se", "tut2017se-evaluation"],
        help="Type of dataset to process.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=10,
        help="Duration of each chunk in seconds (default: 10).",
    )

    args = parser.parse_args()

    # Call the processing function
    process_audio(
        args.ann_folder,
        args.audio_folder,
        args.output_folder,
        args.dataset_type,
        args.chunk_duration,
    )
