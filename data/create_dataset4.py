import pretty_midi
import os
import h5py
import random
from midi_preprocessing import *


##############################################################################
# GLOBAL VARIABLES
##############################################################################


# Define the directory of the input midi file.
INPUT_DIR_PATH = "./raw/maestro-v3.0.0/all"

# Define where the dataset will be saved.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset3/"

# Define the name of the dataset (must end with '.h5').
OUT_FILE_NAME = "all.h5"

##############################################################################



    
def main():

    # Store all midi files in side the input directory.
    print(f"Opening midi files in {INPUT_DIR_PATH}.")
    midi_files = get_midi_from_dir(INPUT_DIR_PATH)
    print(f"\tTotal number of files: {len(midi_files)}.")
    
    # Split the midi files in pieces of 8 bars long.
    print("Splitting midi files...")
    midi_files_splitted = []
    for midi in midi_files:
        midi_splitter(midi, midi_files_splitted, fs=8, n_bars=8)
    print(f"\tTotal number after splitting: {len(midi_files_splitted)}.")

    # Store splitted files.
    # TODO (if needed)

    # Random shuffle. 
    print("Performing random shuffling...")
    random.shuffle(midi_files_splitted)

    # Create the dataset.
    print("Creating dataset...")
    dataset = []
    labels = []
    for i in range(500):
        midi = midi_files_splitted[i]
        get_sample_triplets(midi, dataset, labels, fs=8)
    print(f"\tDataset size: {len(dataset)}.")

    # Data augmentation.
    print("Performing augmentation on dataset...")
    augmented_pairs = []
    augmented_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        for j in range(1, 12):
            augmented_pairs.append(shift_roll_up(sample, j))
            augmented_labels.append(labels[i])
            
    for i in range(len(augmented_pairs)):
        sample = augmented_pairs[i]
        label = augmented_labels[i]
        dataset.append(sample)
        labels.append(label)
    print(f"\tTotal number after augmentation: {len(dataset)}")
    
    with h5py.File(OUT_DIR_PATH + OUT_FILE_NAME, "w") as f:
        f.create_dataset("x", data=dataset, compression = "gzip")
        f.create_dataset("y", data=labels,  compression = "gzip")


if __name__ == "__main__":
    main()
