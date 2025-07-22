import pretty_midi
import os
import h5py
import random
from midi_preprocessing import *


##############################################################################
# GLOBAL VARIABLES
##############################################################################

# Define how many columns of the piano roll correspond to a single bar.
COLS_PER_BAR = 16
# Define the directory of the input midi file.
INPUT_DIR_PATH = "./raw/maestro-v3.0.0/small_tester/"

# Define where the dataset will be saved.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset2/"

# Define the name of the dataset (must end with '.h5').
OUT_FILE_NAME = "to_delete.h5"

##############################################################################


# Define the EMPTY note in a bar
EMPTY = np.zeros(16)

def midi_splitter(midi, out_list, fs=8, n_bars=8):
    # Get piano roll.
    midi_roll = midi.get_piano_roll(fs = fs)    

    # Total number of splits.
    splits = int(midi_roll.shape[1] / (COLS_PER_BAR * n_bars))

    # Total number of columns per split.
    n_cols = COLS_PER_BAR * n_bars

    # Split into samples of size 128xCOLS_PER_BAR.
    for i in range(splits):
        tmp = i * n_cols
        sample = midi_roll[:, tmp:tmp + n_cols]

        # Convert to midi file and store sample.
        sample = piano_roll_to_pretty_midi(sample, fs)
        out_list.append(sample)


def check_octave(bar):
    """
        This function is used to check whether a bar has one shifted note
        out of the admissible range for both the 2 octaves considered
        Args: 
            -bar:  np.array
    """
    # If the exceeded note of the first octave is not empty we fix its position
    if not(np.array_equal(bar[73, :], EMPTY)):
        bar[60,:] = bar[73, :]
        bar[73,:] = EMPTY
    
    # If the exceeded note of the second octave is not empty we fix its position
    if not(np.array_equal(bar[84, :],EMPTY)):
        bar[73,:] = bar[84, :]
        bar[84,:] = EMPTY


def shift_roll_up(sample, n):
    """
        This function can be used to circularly shift by n semitones the 2 bars
        checking for their integrity after the shift
        Args:
            -sample: (np.array, np.array)
            -n:      int
    """
    # Getting the 2 bar contained in a sample (previous and current)
    prev, curr = sample

    # Iterating through all values in the shift range
    for i in range(n):
        # Adding one fake note at the beginning of the roll
        prev = np.vstack((EMPTY,prev))
        # Copying last note in the first one
        prev[0,:] = prev[128,:]
        # Deleting the last note (we have an excess of notes by one)
        prev = np.delete(prev, 128, 0)
        
        # Adding one fake note at the beginning of the roll
        curr = np.vstack((EMPTY,curr))
        # Copying last note in the first one
        curr[0,:] = curr[128,:]
        # Deleting the last note (we have an excess of notes by one)
        curr = np.delete(curr, 128, 0)
        
        # Check on prev and curr for both the last note of the 2 considered octaves
        check_octave(prev)
        check_octave(curr)

    # Transform in bool.
    prev = prev.astype(bool)
    curr = curr.astype(bool)

    return prev,curr


def get_sample_pairs(midi, out_list, fs=8):
        # Define parameters.
        #bar_duration = get_bar_duration(midi)
        #fs = COLS_PER_BAR / bar_duration
        
        # Create meoldy piano roll.
        melody_roll = extract_melody(midi, fs=fs)
     
        # Binarize melody piano roll.
        melody_roll[melody_roll > 0] = 1
        melody_roll = melody_roll.astype(bool)

        # Normalize melody piano roll.
        melody_roll = normalize_melody_roll(melody_roll, lb=60, ub=83)
    
        # Add zero padding to the end of the piano roll (if necessary).
        zero_padding = melody_roll.shape[1] % COLS_PER_BAR
        melody_roll = np.pad(
            melody_roll, ((0, 0), (0, zero_padding)),
            mode='constant', constant_values=0
        )

        # Fill piano roll, i.e. remove pauses.
        melody_roll = fill_melody_pauses(melody_roll)

        # Split into samples of size 128xCOLS_PER_BAR matrices.
        splitted = []
        splits = int(melody_roll.shape[1] / COLS_PER_BAR)
        for i in range(splits):
            tmp = i * COLS_PER_BAR
            sample = melody_roll[:, tmp:tmp+COLS_PER_BAR]
            splitted.append(sample)

        # Create the pairs of previous and current bars.
        for i in range(1, len(splitted)):
            pair = splitted[i-1], splitted[i]
            out_list.append(pair)


def get_midi_from_dir(dir_path):
    midi_files = []

    # Iterate through all the files.
    for filename in os.listdir(dir_path):
        # Skip non-midi files.
        if not (filename.endswith(".midi") or filename.endswith(".mid")):
            continue
 
        # Open the midi file.
        midi = pretty_midi.PrettyMIDI(os.path.join(dir_path, filename))
        midi_files.append(midi)

    return midi_files

    
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
    for midi in midi_files_splitted:
        get_sample_pairs(midi, dataset, fs=8)
    print(f"\tDataset size: {len(dataset)}.")

    # Data augmentation.
    print("Performing augmentation on dataset...")
    augmented = []
    for sample in dataset:
        for i in range(1, 12):
            augmented.append(shift_roll_up(sample, i))
    for sample in augmented:
        dataset.append(sample)
    print(f"\tTotal number after augmentation: {len(dataset)}")

    # Store the dataset.
    with h5py.File(OUT_DIR_PATH + OUT_FILE_NAME, "w") as f:
        f.create_dataset("x", data=dataset, compression="gzip")

if __name__ == "__main__":
    main()
