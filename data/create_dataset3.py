import pretty_midi
import os
import h5py
from midi_preprocessing import *


### GLOBAL VARIABLES ###
# Define the number of cols per bar.
COLS_PER_BAR = 16
# Define the directory of the dataset.
INPUT_DIR_PATH =[
    "./raw/maestro-v3.0.0/splitted_2018",
#    "./raw/maestro-v3.0.0/2017/",
#    "./raw/maestro-v3.0.0/2015/",
#    "./raw/maestro-v3.0.0/2014/"
]

# Define the output directory path.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset2/"
# Define output file name.
OUT_FILE_NAME = "dataset.h5"


def extract_samples(dir_path, samples, labels):
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

        # Applying selection in midi dataset
        """
        if not midi_selection(midi, fs):
            print("skipped", filename)
            continue
        """
        
        # Create meoldy piano roll.
        melody_roll = extract_melody(midi, fs=fs)
        piano_roll = midi.get_piano_roll(fs = fs)
        
        # Binarize melody piano roll.
        melody_roll[melody_roll > 0] = 1
        melody_roll = melody_roll.astype(bool)
        
        piano_roll[piano_roll > 0] = 1
        piano_roll = piano_roll.astype(bool)

        # Normalize melody piano roll.
        melody_roll = normalize_melody_roll(melody_roll, lb=60, ub=83)

        # Applying selection based on normalized melody 
        """
        if not melody_selection(melody_roll): 
            print("skipped", filename)
            continue
        """
    
        # Add zero padding to the end of the piano roll.
        zero_padding = (melody_roll.shape[1] % COLS_PER_BAR)
        melody_roll = np.pad(melody_roll, ((0, 0), (0, zero_padding)), mode='constant', constant_values=0)
        piano_roll = np.pad(piano_roll, ((0, 0), (0, zero_padding)), mode='constant', constant_values=0)

        # Fill piano roll.
        melody_roll = fill_melody_pauses(melody_roll)

        print("Splitting sample ->", filename)

        # Finding Main chords of the bars
        
    
        # Split into samples of size 128xCOLS_PER_BAR.
        splitted = []
        splits = int(melody_roll.shape[1] / COLS_PER_BAR)
        for i in range(splits):
            tmp = i * COLS_PER_BAR
            sample = melody_roll[:, tmp:tmp+COLS_PER_BAR]
            splitted.append(sample)

        # Create the pair of previous and current bars.
        for i in range(1, len(splitted)):
            main_chord = main_chords(splitted[i]).astype(bool)
            labels.append(main_chord)
            pair = splitted[i-1], splitted[i]
            samples.append(pair)

        print(f"\tSplitted in {splits} samples.")

    
def main():
    # Store the samples in here.
    samples = []
    labels = []
    
    for dir in INPUT_DIR_PATH:
        extract_samples(dir, samples, labels)

    print("Dataset has", len(samples), " samples")

    # Store the dataset in a .h5 file.
    #print("Dataset reduced to", len(samples), " samples")
    samples = np.stack(samples)
    with h5py.File(OUT_DIR_PATH + OUT_FILE_NAME, "w") as f:
        f.create_dataset("x", data=samples, compression = "gzip")
        f.create_dataset("y", data=labels,  compression = "gzip")


if __name__ == "__main__":
    main()
