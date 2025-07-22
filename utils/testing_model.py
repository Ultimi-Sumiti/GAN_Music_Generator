from data.midi_preprocessing import *
from utils.dataset_loader import *
import random
import pygame
import time

def reproduce_midi(file_midi: str):
    """
    Riproduce un file MIDI utilizzando pygame.

    Args:
        file_midi (str): path to the midi file to play.
    """
    try:
        # Initialize pygame mixer
        # These values are standard: frequency, bit, channels, buffer size
        pygame.mixer.init(44100, -16, 2, 512)

        pygame.init()

        print(f"🎵 Reproduction of'{file_midi}'...")

        # Load and play music
        pygame.mixer.music.load(file_midi)
        pygame.mixer.music.play()

        # Wait untile the song is fully played
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Make sure the file loaded is valid")
    except FileNotFoundError:
        print(f"Error: File not found at the given path '{file_midi}'")
    finally:
        # Clean and close the mixer
        if pygame.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            pygame.quit()
        print("Reproduction ended.")


# Generate a melody. Returns the pretty midi file.
def generate_melody(model, n_bars, dataset, verbose=False):
    # Set model in evaluation.
    model.eval()
    model = model.cpu()
    
    # Generate noise.
    noise = torch.randn(n_bars, 1, 100)

    # Random index.
    rnd_idx = random.randint(0, len(dataset)-1)
    
    # Choose the first random sample from the dataset.
    bar_0, _ = dataset[rnd_idx] # [1, 128, 16]
    bar_0 = bar_0.unsqueeze(0)  # [1, 1, 128, 16]

    # Generate 8 bar.
    bars = [bar_0]
    for i, z in enumerate(noise):
        # Previous bar.
        prev = bars[i]
    
        # Create the pair.
        x = z, prev
    
        # Generate current bar.
        curr = model(x)
    
        # Save genjerated bar
        bars.append(curr)

    
    # Convert bars in numpy array.
    bars_numpy = []
    for bar in bars:
        bar = bar.squeeze(0, 1).detach().numpy()
        bars_numpy.append(bar)
    
    # Create the full piano roll.
    full_piano_roll = np.hstack([bar for bar in bars_numpy])
    if verbose:
        print("Full piano roll")
        print("Shape:", full_piano_roll.shape)
        show_piano_roll(full_piano_roll)
    
    # Multiply by 50.
    full_piano_roll *= 50

    # Create midi file.
    pm = piano_roll_to_pretty_midi(full_piano_roll, fs=8)
    
    return pm

# Generate a melody accompanied of chords
def generate_melody_and_chords(model, n_bars, dataset, verbose=False):
# Set model in evaluation.
    model.eval()
    model = model.cpu()

    # Generate noise.
    noise = torch.randn(n_bars - 1, 1, 100)

    # Random index.
    rnd_idx = random.randint(0, len(dataset)-1)

    # Choose the first random sample from the dataset.
    bar_0, _ , _ = dataset[rnd_idx] # [1, 128, 16]
    bar_0 = bar_0.unsqueeze(0)  # [1, 1, 128, 16]

    _ , _ , chord_0 = dataset[rnd_idx] # [13 , 1, 1]
    chord_0 = chord_0.unsqueeze(0) # [1, 13 , 1 , 1] 

    # Generate 8 bar.
    bars = [bar_0]
    for i , z in enumerate(noise):
        # Previous bar.
        prev = bars[i]

        # Create the triplet.
        x = z, prev, chord_0

        # Generate current bar.
        curr = model(x)

        # Save genjerated bar
        bars.append(curr)

    # Processing to add in a specific way the coding of the chord for each bar
    #chord_0 = torch.zeros((1,13,1,1), dtype=int)
    #chord_0[0,1,0,0] = 1
    decoding_chord(bars,chord_0)

    # Convert bars in numpy array.
    bars_numpy = []
    for bar in bars:
        bar = bar.squeeze(0, 1).detach().numpy()
        bars_numpy.append(bar)

    # Create the full piano roll.
    full_piano_roll = np.hstack([bar for bar in bars_numpy])
    if verbose:
        print("Full piano roll")
        print("Shape:", full_piano_roll.shape)
        show_piano_roll(full_piano_roll)

    # Multiply by 50.
    full_piano_roll *= 50

    # Create midi file.
    pm = piano_roll_to_pretty_midi(full_piano_roll, fs=8)

    return pm

MAJOR_CHORD_NOTE_TRIPLETS = [
    [0, 4, 7],    # C Major (Tonic 0)
    [1, 5, 8],    # C# Major (Tonic 1)
    [2, 6, 9],    # D Major (Tonic 2)
    [3, 7, 10],   # D# Major (Tonic 3)
    [4, 8, 11],   # E Major (Tonic 4)
    [5, 9, 0],    # F Major (Tonic 5)
    [6, 10, 1],   # F# Major (Tonic 6)
    [7, 11, 2],   # G Major (Tonic 7)
    [8, 0, 3],    # G# Major (Tonic 8)
    [9, 1, 4],    # A Major (Tonic 9)
    [10, 2, 5],   # A# Major (Tonic 10)
    [11, 3, 6]    # B Major (Tonic 11)
]

MINOR_CHORD_NOTE_TRIPLETS = [
    [9, 0, 4],    # A Minor (Tonic 9)
    [10, 1, 5],   # A# Minor (Tonic 10)
    [11, 2, 6],   # B Minor (Tonic 11)
    [0, 3, 7],    # C Minor (Tonic 0)
    [1, 4, 8],    # C# Minor (Tonic 1)
    [2, 5, 9],    # D Minor (Tonic 2)
    [3, 6, 10],   # D# Minor (Tonic 3)
    [4, 7, 11],   # E Minor (Tonic 4)
    [5, 8, 0],    # F Minor (Tonic 5)
    [6, 9, 1],    # F# Minor (Tonic 6)
    [7, 10, 2],   # G Minor (Tonic 7)
    [8, 11, 3]    # G# Minor (Tonic 8)
]

# Function to insert a Basic pattern chords 
def pattern_chords_1(bars, selected_chord, offset, n_bars):
    for elem,b in enumerate(bars):
                #print(b.shape[2] - 1 - offset - 12)
                #print(b.shape[2] - 1 - offset)
                
                if elem < n_bars:
                    # row minor
                    for i in range(offset, 12 + offset, 1):
                        for j in range(b.shape[3] - 1):
                            #print(i)
                            #print(selected_chord[0],selected_chord[1], selected_chord[2])
                            if i == selected_chord[0] + offset or i == selected_chord[1] + offset or i == selected_chord[2] + offset:
                                b[0,0,i,j] = 0.75


# Function to decoding the chords emmbedded in the output tensor generated by the network
def decoding_chord(bars, chord_0):
    index = -1
     # Major chord
    if chord_0[0,12,0,0] == 0:
        print("Major chords")
        encoded_chord = np.where(chord_0 == 1)

        if encoded_chord[0].size > 0:
            index = encoded_chord[1][0]
        
        selected_chord = MAJOR_CHORD_NOTE_TRIPLETS[index]
        # To select the octave of the chord
        offset = 4
        offset = offset * 12
        
        pattern_chords_1(bars, selected_chord, offset, 7)

    # Minor chord
    elif chord_0[0,12,0,0] == 1:
        print("Minors chords")
        encoded_chord = np.where(chord_0 == 1)

        if encoded_chord[0].size > 0:
            index = encoded_chord[1][0]
        
        selected_chord = MINOR_CHORD_NOTE_TRIPLETS[index]
        # To select the octave of the chord
        offset = 4
        offset = offset * 12

        pattern_chords_1(bars,selected_chord,offset, 7)
    