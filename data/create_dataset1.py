import pretty_midi
import os
import h5py

from midi_preprocessing import *


### GLOBAL VARIABLES ###
# Define the number of cols per bar.
COLS_PER_BAR = 16
# Define the directory of the dataset.
INPUT_DIR_PATH = "./raw/maestro-v3.0.0/2018"
# Define the output directory path.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset1/"
# Define output file name.
OUT_FILE_NAME = "2018_50k_augmented.h5"


def extract_samples(dir_path, samples):
    # Iterate through.
    for filename in os.listdir(dir_path):
    
        # Skip non-midi files.
        if not filename.endswith(".midi"):
            continue
    
        # Open the midi file.
        midi = pretty_midi.PrettyMIDI(os.path.join(dir_path, filename))
    
        # Define parameters.
        bar_duration = get_bar_duration(midi)
        fs = COLS_PER_BAR / bar_duration
        
        # Create meoldy piano roll.
        melody_roll = extract_melody(midi, fs=fs)
     
        # Binarize melody piano roll.
        melody_roll[melody_roll > 0] = 1
        melody_roll = melody_roll.astype(bool)

        # Normalize melody piano roll.
        melody_roll = normalize_melody_roll(melody_roll, lb=60, ub=83)
    
        # Add zero padding to the end of the piano roll.
        zero_padding = COLS_PER_BAR - (melody_roll.shape[1] % COLS_PER_BAR)
        melody_roll = np.pad(melody_roll, ((0, 0), (0, zero_padding)), mode='constant', constant_values=0)

        # Fill piano roll.
        melody_roll = fill_melody_pauses(melody_roll)

        print("Splitting sample ->", filename)
    
        # Split into samples of size 128xCOLS_PER_BAR.
        splits = int(melody_roll.shape[1] / COLS_PER_BAR)
        for i in range(splits):
            tmp = i * COLS_PER_BAR
            sample = melody_roll[:, tmp:tmp+COLS_PER_BAR]
            samples.append(sample)

        print(f"\tSplitted in {splits} samples.")

    
    #print(len(samples))
    #samples = np.stack(samples)
    #with h5py.File("midi_dataset.h5", "w") as f:
    #    f.create_dataset("midi_matrices", data=samples, compression="gzip")


def main():
    # Store the samples in here.
    samples = []
    
    extract_samples(INPUT_DIR_PATH, samples)

    print("Dataset has", len(samples), " samples")

    # Store the dataset in a .h5 file.
    samples = np.stack(samples)
    with h5py.File(OUT_DIR_PATH + OUT_FILE_NAME, "w") as f:
        f.create_dataset("x", data=samples, compression="gzip")


if __name__ == "__main__":
    main()
