from midi_preprocessing import *
import pretty_midi
import os
import h5py

# Define the number of cols per bar.
COLS_PER_BAR = 16

# Define the directory of the dataset.
INPUT_DIR_PATH = "./raw/maestro-v3.0.0/2018/"

# Define the output directory path.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset1/"

# Create the output dir if necessary.
if not os.path.exists(OUT_DIR_PATH):
    os.makedirs(OUT_DIR_PATH)

# Join path.
#dir_path = os.path.join(INPUT_DIR_PATH, dir_path)
#dir_path = os.path.join(INPUT_DIR_PATH)


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
    
        # Add zero padding to the end of the piano roll.
        zero_padding = COLS_PER_BAR - (melody_roll.shape[1] % COLS_PER_BAR)
        melody_roll = np.pad(melody_roll, ((0, 0), (0, zero_padding)), mode='constant', constant_values=0)
        
        # Fill piano roll.
        melody_roll = fill_melody_pauses(melody_roll)

        print("Splitting file ->", filename)
    
        # Split into samples of size 128xCOLS_PER_BAR.
        for i in range(int(melody_roll.shape[1] / COLS_PER_BAR)):
            tmp = i * COLS_PER_BAR
            sample = melody_roll[:, tmp:tmp+COLS_PER_BAR]
            samples.append(sample)
    
    #print(len(samples))
    #samples = np.stack(samples)
    #with h5py.File("midi_dataset.h5", "w") as f:
    #    f.create_dataset("midi_matrices", data=samples, compression="gzip")


def main():
    # Store the samples in here.
    samples = []
    
    extract_samples(INPUT_DIR_PATH, samples)

    print("Creating a dataset with ")

    # Store the dataset in a .h5 file.
    samples = np.stack(samples)
    with h5py.File(OUT_DIR_PATH + "d1.h5", "w") as f:
        f.create_dataset("x", data=samples, compression="gzip")



if __name__ == "__main__":
    main()

# THE OLD STRATEGY - TO REMOVE.
## Create the dataset.
#for dir_path in os.listdir(INPUT_DIR_PATH):
#
#    # Join path.
#    dir_path = os.path.join(INPUT_DIR_PATH, dir_path)
#
#    for filename in os.listdir(dir_path):
#
#        # Consider only midi files.
#        if not filename.endswith(".midi"):
#            continue
#
#        # Open the midi file.
#        midi = pretty_midi.PrettyMIDI(os.path.join(dir_path, filename))
#    
#        # Define parameters.
#        bar_duration = get_bar_duration(midi)
#        fs = COLS_PER_BAR / bar_duration
#        
#        # Create meoldy piano roll.
#        melody_roll = extract_melody(midi, fs=fs)
#     
#        # Binarize melody piano roll.
#        melody_roll[melody_roll > 0] = 1
#    
#        # Add zero padding to the end of the piano roll.
#        zero_padding = COLS_PER_BAR - (melody_roll.shape[1] % COLS_PER_BAR)
#        melody_roll = np.pad(melody_roll, ((0, 0), (0, zero_padding)), mode='constant', constant_values=0)
#        
#        # Fill piano roll.
#        melody_roll = fill_melody_pauses(melody_roll)
#    
#        # Save the sample
#        fn = os.path.splitext(filename)[0] + ".pt"
#        out_file_path = OUT_DIR_PATH + os.path.splitext(filename)[0] + ".pt"
#        print("Creating ->", out_file_path)
#
#    
#        # Convert to tensor.
#        #melody_roll = torch.from_numpy(melody_roll)
#        #torch.save(melody_roll, out_file_path)
#    
#        # Split into samples of size 128xCOLS_PER_BAR.
#        samples = []
#        for i in range(int(melody_roll.shape[1] / COLS_PER_BAR)):
#            tmp = i * COLS_PER_BAR
#            samples.append(melody_roll[:, tmp:tmp+COLS_PER_BAR])
#    
#        for i, sample in enumerate(samples):
#            fn = os.path.splitext(filename)[0] + str(i) + ".pt"
#            out_file_path = out_dir_path + fn
#            #np.savetxt(out_file_path, melody_roll, delimiter=",", fmt="%.6f")
#            # Convert to tensor.
#            sample = torch.from_numpy(sample)
#            torch.save(sample, out_file_path)

#filename = out_dir_path + "MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--3.pt"
#tensor = torch.load(filename)
#print(tensor.shape)
#print((tensor > 0).sum().item())
