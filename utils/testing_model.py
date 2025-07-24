""" In this class we have all the function and classes related to the testing phase. """

from data.midi_preprocessing import *
from utils.dataset_loader import *
import random
import pygame
import time

# CONSTANTS:

# The following "constants" are used to identidy in the octave the major and minor
# chords triplets.
MAJOR_CHORD_NOTE_TRIPLETS = [
    [0, 4, 7],  # C Major (Tonic 0)
    [1, 5, 8],  # C# Major (Tonic 1)
    [2, 6, 9],  # D Major (Tonic 2)
    [3, 7, 10],  # D# Major (Tonic 3)
    [4, 8, 11],  # E Major (Tonic 4)
    [5, 9, 0],  # F Major (Tonic 5)
    [6, 10, 1],  # F# Major (Tonic 6)
    [7, 11, 2],  # G Major (Tonic 7)
    [8, 0, 3],  # G# Major (Tonic 8)
    [9, 1, 4],  # A Major (Tonic 9)
    [10, 2, 5],  # A# Major (Tonic 10)
    [11, 3, 6],  # B Major (Tonic 11)
]

MINOR_CHORD_NOTE_TRIPLETS = [
    [9, 0, 4],  # A Minor (Tonic 9)
    [10, 1, 5],  # A# Minor (Tonic 10)
    [11, 2, 6],  # B Minor (Tonic 11)
    [0, 3, 7],  # C Minor (Tonic 0)
    [1, 4, 8],  # C# Minor (Tonic 1)
    [2, 5, 9],  # D Minor (Tonic 2)
    [3, 6, 10],  # D# Minor (Tonic 3)
    [4, 7, 11],  # E Minor (Tonic 4)
    [5, 8, 0],  # F Minor (Tonic 5)
    [6, 9, 1],  # F# Minor (Tonic 6)
    [7, 10, 2],  # G Minor (Tonic 7)
    [8, 11, 3],  # G# Minor (Tonic 8)
]


# FUNCTIONS:


def reproduce_midi(file_midi: str):
    """This function allow to reproduce the file midi in the file path given as
    argument.

    Arguments:
        file_midi (str): path to the midi file to play.
    """

    try:
        # Initialize pygame mixer.
        # These values are standard: frequency, bit, channels, buffer size.
        pygame.mixer.init(44100, -16, 2, 512)

        pygame.init()

        print(f"🎵 Reproduction of'{file_midi}'...")

        # Load and play music.
        pygame.mixer.music.load(file_midi)
        pygame.mixer.music.play()

        # Wait untile the song is fully played.
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Make sure the file loaded is valid!")
    except FileNotFoundError:
        print(f"Error: File not found at the given path '{file_midi}'!")
    finally:
        # Clean and close the mixer.
        if pygame.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            pygame.quit()
        print("Reproduction ended.")


# Generate a melody. Returns the pretty midi file.
def generate_melody(model, n_bars, dataset, verbose=False):
    """This function is used to generate a file midi given a model for the generation, a
    number of bars and an input dataset.

    Arguments:
        model     This is the model used for generation.
        n_bars    This is the amount of bars to generate.
        dataset   This variable is the datset used.
        verbose   This boolean variable tells whether the generation should be verbose or not.
    """

    # Set model in evaluation.
    model.eval()
    model = model.cpu()

    # Generate noise.
    noise = torch.randn(n_bars - 1, 1, 100)

    # Random index.
    rnd_idx = random.randint(0, len(dataset) - 1)

    # Choose the first random sample from the dataset.
    bar_0, _ = dataset[rnd_idx]  # [1, 128, 16]
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

        # Save generated bar
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


def generate_melody_v3(model, n_bars, dataset, verbose=False):
    """This function returns the bars and the chord, so using in combination with decoding_chord
    function.
    We can decide the pattern based on chord_0 to add to this melody.
    Generate a melody that in future processing can be accompanied with chords.

    Argsuments:
        model     This is the model used for generation.
        n_bars    This is the amount of bars to generate.
        dataset   This variable is the datset used.
        verbose   This boolean variable tells whether the generation should be verbose or not.
    """
    # Set model in evaluation.
    model.eval()
    model = model.cpu()

    # Generate noise.
    noise = torch.randn(n_bars - 1, 1, 100)

    # Random index.
    rnd_idx = random.randint(0, len(dataset) - 1)

    # Choose the first random sample from the dataset.
    bar_0, _, _ = dataset[rnd_idx]  # [1, 128, 16]
    bar_0 = bar_0.unsqueeze(0)  # [1, 1, 128, 16]

    _, _, chord_0 = dataset[rnd_idx]  # [13 , 1, 1]
    chord_0 = chord_0.unsqueeze(0)  # [1, 13 , 1 , 1]

    # Generate 8 bar.
    bars = [bar_0]
    for i, z in enumerate(noise):
        # Previous bar.
        prev = bars[i]

        # Create the triplet.
        x = z, prev, chord_0

        # Generate current bar.
        curr = model(x)

        # Save genjerated bar.
        bars.append(curr)

    pm = bars_to_piano_roll(bars, verbose)

    return pm, bars, chord_0


def bars_to_piano_roll(bars, verbose):
    """This function converts bar tensors (list of bar tensors) to pretty midi objects
    and print them based on the boolean.

    Arguments:
        bars      This is a list of bar tensors to convert to piano rolls.
        verbose   This boolean variable teels whether we have to show the piano rolls or not.
    """
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


# Function to decoding the chords emmbedded in the output tensor generated by the network
def decoding_chord(bars, chord_0, pattern_choice, offset=4, n_bars=4):
    """ This function allow to decode chords embedded in the output tensore generated 
    by the network.

    Arguments:
        bars             This is a list of bar tensors. 
        chord_0          This is the initial chord.
        pattern_choice   This parameter allow to decide which pattern for the chords we should use.
        offset           The octave where we play the chord. 
        n_bars           This is the amount of bars we should play the chords for.
    """
    
    index = -1
    
    # Major chord.
    if chord_0[0, 12, 0, 0] == 0:
        print("Major chords")
        encoded_chord = np.where(chord_0 == 1)

        if encoded_chord[0].size > 0:
            index = encoded_chord[1][0]

        selected_chord = MAJOR_CHORD_NOTE_TRIPLETS[index]
        
        # To select the octave of the chord.
        offset = 4
        offset = offset * 12
        #
        pattern_chords(bars, selected_chord, pattern_choice, offset, n_bars)

    # Minor chord.
    elif chord_0[0, 12, 0, 0] == 1:
        print("Minors chords")
        encoded_chord = np.where(chord_0 == 1)

        if encoded_chord[0].size > 0:
            index = encoded_chord[1][0]

        selected_chord = MINOR_CHORD_NOTE_TRIPLETS[index]
        
        # To select the octave of the chord.
        offset = offset * 12

        pattern_chords(bars, selected_chord, pattern_choice, offset, n_bars)

    

def pattern_chords(bars, selected_chord, choice = 1, offset = 4, n_bars = 4):
    """ This function is used to insert a Basic pattern chords, chosing the one
    with the param choice.
    Arguments: 
        bars
        selected_chord   This is the chord we selected.
        choice           This the choice of the pattern to use.
        offset           This is the octave we play the chord in.
        n_bars           This is the number of bars we need to process.
    """
    # Choosing the corresponding patterns.
    match choice:
      case 1:
           print("selected 1")
           return pattern_1(bars, selected_chord, offset, n_bars)
      case 2:
           print("selected 2")
           return pattern_2(bars, selected_chord, offset, n_bars)
      case 3:
           print("selected 3")
           return pattern_3(bars, selected_chord, offset)
      case 4: 
           print("selected 4")
           return pattern_4(bars, selected_chord, offset, n_bars) 
      case 5:
           print("selected 5")
           return pattern_5(bars, selected_chord, offset) 
      case 6:
           print("selected 6")
           return pattern_6(bars, selected_chord, offset)

# ----------- PATTERNS FOR CHORDS -----------
# n chord of a specific octave, setted with the offset.
def pattern_1(bars, selected_chord, offset, n_bars):
    """This function implements the first chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """
    for elem, b in enumerate(bars):
        # print(b.shape[2] - 1 - offset - 12)
        # print(b.shape[2] - 1 - offset)
        if elem < n_bars:
            # Row minor (needed to iterate in the correct block of a single bar (elem))
            for i in range(offset, 12 + offset, 1):
                for j in range(b.shape[3] - 1):
                    # print(i)
                    # print(selected_chord[0],selected_chord[1], selected_chord[2])
                    if (
                        i == selected_chord[0] + offset
                        or i == selected_chord[1] + offset
                        or i == selected_chord[2] + offset
                    ):
                        b[0, 0, i, j] = 0.7


# n chords of an octave and len(bars) - n chords of an octave below.
def pattern_2(bars, selected_chord, offset, n_bars):
    """This function implements the second chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """
    for elem, b in enumerate(bars):
        if elem < n_bars:
            # Row minor.
            for i in range(offset, 12 + offset, 1):
                for j in range(b.shape[3] - 1):
                    # Activation of the chord.
                    if (
                        i == selected_chord[0] + offset
                        or i == selected_chord[1] + offset
                        or i == selected_chord[2] + offset
                    ):
                        b[0, 0, i, j] = 0.7

        if elem <= len(bars) and elem >= n_bars:
            for i in range(offset, 12 + offset, 1):
                for j in range(b.shape[3] - 1):
                    # Activation of the chord.
                    if (
                        i == selected_chord[0] + offset
                        or i == selected_chord[1] + offset
                        or i == selected_chord[2] + offset
                    ):
                        b[0, 0, i - 12, j] = 0.7


# n chords of an octave and len(bars) - n chords of an octave below.
def pattern_3(bars, selected_chord, offset):
    """This function implements the third chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """
    for elem, b in enumerate(bars):
        if elem <= len(bars) and elem % 2 == 0:
            # Row minor.
            for i in range(offset, 12 + offset, 1):
                for j in range(b.shape[3] - 1):
                    # Activation of the chord.
                    if (
                        i == selected_chord[0] + offset
                        or i == selected_chord[1] + offset
                        or i == selected_chord[2] + offset
                    ):
                        b[0, 0, i, j] = 0.7

        if elem <= len(bars) and elem % 2 == 1:
            for i in range(offset, 12 + offset, 1):
                for j in range(b.shape[3] - 1):
                    # Activation of the chord.
                    if (
                        i == selected_chord[0] + offset
                        or i == selected_chord[1] + offset
                        or i == selected_chord[2] + offset
                    ):
                        b[0, 0, i - 12, j] = 0.7


# Make the chord press for tunable number of steps and silence steps for 3 time in a bar.
# directly in this function is possible to tune these pars.
def pattern_4(bars, selected_chord, offset, n_bars):
    """This function implements the fourth chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """

    for elem, b in enumerate(bars):
        if elem % n_bars == 0:
            time_idx = 0
            while time_idx < b.shape[3]:

                # Chord for 4 time step (tunable).
                for _ in range(2):
                    if time_idx < b.shape[3]:
                        for note_offset in selected_chord:
                            note_pitch = note_offset + offset
                            if 0 <= note_pitch < b.shape[2]:
                                b[0, 0, note_pitch, time_idx] = 0.7
                        time_idx += 1
                    else:
                        break

                # Pause for 3 time steps (tunable).
                time_idx += 3

                # Chord for 2 time step (tunable).
                for _ in range(3):
                    if time_idx < b.shape[3]:
                        for note_offset in selected_chord:
                            note_pitch = note_offset + offset
                            if 0 <= note_pitch < b.shape[2]:
                                b[0, 0, note_pitch, time_idx] = 0.7
                        time_idx += 1
                    else:
                        break

                # Pause for 1 time step (tunable).
                time_idx += 1

                # Chord until the end of the bar.
                while time_idx < b.shape[3]:
                    for note_offset in selected_chord:
                        note_pitch = note_offset + offset
                        if 0 <= note_pitch < b.shape[2]:
                            b[0, 0, note_pitch, time_idx] = 0.7
                    time_idx += 1


# Equal to pattern 4 but similary to pattern 3 alternates every bar with a different octave.
def pattern_5(bars, selected_chord, offset):
    """This function implements the fifth chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """
    for elem, b in enumerate(bars):
        current_offset = offset
        if elem % 2 != 0:
            current_offset = offset - 12

        time_idx = 0
        while time_idx < b.shape[3]:
            # Chord for 4 time step (tunable).
            for _ in range(2):
                if time_idx < b.shape[3]:
                    for note_offset in selected_chord:
                        note_pitch = note_offset + current_offset
                        if 0 <= note_pitch < b.shape[2]:
                            b[0, 0, note_pitch, time_idx] = 0.9
                    time_idx += 1
                else:
                    break

            # Pause for 3 time steps (tunable).
            time_idx += 3

            # Chord for 2 time step (tunable).
            for _ in range(5):
                if time_idx < b.shape[3]:
                    for note_offset in selected_chord:
                        note_pitch = note_offset + current_offset
                        if 0 <= note_pitch < b.shape[2]:
                            b[0, 0, note_pitch, time_idx] = 0.7
                    time_idx += 1
                else:
                    break

            # Pause for 1 time steps (tunable).
            time_idx += 1

            # Chord until the end of the bar.
            while time_idx < b.shape[3]:
                for note_offset in selected_chord:
                    note_pitch = note_offset + current_offset
                    if 0 <= note_pitch < b.shape[2]:
                        b[0, 0, note_pitch, time_idx] = 0.9
                time_idx += 1


# Similar to pattern 5 but in the odd bars we add some different processing/pattern.
def pattern_6(bars, selected_chord, offset):
    """This function implements the sixth chords pattern.

    Arguments:
        bars             This is the bars where we place the chords.
        selected_chord   This is the chord selected to be reproduced.
        offset           This is the octave where we play the chords.
        n_bars           This is the amount of bars where we play the chords.
    """
    for elem, b in enumerate(bars):
        current_offset = offset

        # For even bars.
        if elem % 2 == 0:
            # Equal to pattern_5.
            time_idx = 0
            while time_idx < b.shape[3]:
                # Chord for 2 time step.
                for _ in range(2):
                    if time_idx < b.shape[3]:
                        for note_offset in selected_chord:
                            note_pitch = note_offset + current_offset
                            if 0 <= note_pitch < b.shape[2]:
                                b[0, 0, note_pitch, time_idx] = 0.9
                        time_idx += 1
                    else:
                        break

                # Pause for 3 time steps.
                time_idx += 3

                # Chord for 5 time step.
                for _ in range(5):
                    if time_idx < b.shape[3]:
                        for note_offset in selected_chord:
                            note_pitch = note_offset + current_offset
                            if 0 <= note_pitch < b.shape[2]:
                                b[0, 0, note_pitch, time_idx] = 0.7
                        time_idx += 1
                    else:
                        break

                # Pause for 1 time step.
                time_idx += 1

                # Chord until the end of the bar.
                while time_idx < b.shape[3]:
                    for note_offset in selected_chord:
                        note_pitch = note_offset + current_offset
                        if 0 <= note_pitch < b.shape[2]:
                            b[0, 0, note_pitch, time_idx] = 0.9
                    time_idx += 1

        # For odd bars.
        else:
            current_offset = offset - 12
            start_idx = b.shape[3] - 6

            # Resetting all other values to 0 before setting the last 4 beats for clarity.
            # b[0, 0, :, :] = 0.0 # Optional: uncomment if you want to clear the bar first.
            for time_idx in range(start_idx, b.shape[3] - 2):
                for note_offset in selected_chord:
                    note_pitch = note_offset + current_offset
                    if 0 <= note_pitch < b.shape[2]:
                        b[0, 0, note_pitch, time_idx] = 0.9
