import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import os

# Max number of piano notes.
MAX_PIANO_NOTES = 20
# Min number of piano notes.
MIN_PIANO_NOTES = 5
# Major chords first notes (starts from C = 1th note).
MAJOR_CHORDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Minor chords first notes (starts from A = 10th note).
MINOR_CHORDS = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
# Define how many columns of the piano roll correspond to a single bar.
COLS_PER_BAR = 16
# Define the EMPTY note in a bar.
EMPTY = np.zeros(16)


def fill_melody_pauses(piano_roll):
    """This function is used to fill the pauses present in the piano roll.

    Arguments:
        piano_roll   This a simple piano roll object.
    """
    # The filled piano roll.
    filled_roll = piano_roll.copy()

    # Find the first active note.
    first_active_pitch = 0
    first_active_t = 0
    for t in range(filled_roll.shape[1]):
        if np.any(filled_roll[:, t]):
            first_active_pitch = np.argmax(filled_roll[:, t])
            first_active_t = t
            break

    # Fill pause in the roll.
    last_active_pitch = None
    for t in range(filled_roll.shape[1]):
        if np.any(filled_roll[:, t]):
            last_active_pitch = np.argmax(filled_roll[:, t])
        elif last_active_pitch is not None:
            filled_roll[last_active_pitch, t] = filled_roll[last_active_pitch, t - 1]
        else:
            filled_roll[first_active_pitch, t] = filled_roll[
                first_active_pitch, first_active_t
            ]

    return filled_roll


def normalize_melody_roll(piano_roll, lb, ub):
    """This function is used to normalize the melody (only one note per time step)
    by getting a lower bound (lb) and and an upper bound (ub) for notes position 
    in the 0-127 interval.

    Arguments: 
        piano_rool The input piano roll object. 
        lb         This is the lower bound (integer within 0-127 interval)
        ub         This is the upper bound (integer within 0-127 interval)
    """
    
    assert lb <= ub, "Lower bound must be smaller or equal than upper bound."
    assert lb >= 0, "Lower bound must be bigger or equal to 0."
    assert lb <= 127 "Lower bound must be smaller or equal to 127."
    assert ub >= 0, "Upper bound must be bigger or equal to 0."
    assert ub <= 127 "Upper bound must be smaller or equal to 127."
    
    # Output roll.
    normalized_piano_roll = np.zeros_like(piano_roll)
    
    # Finds coordinates of active notes.
    active_pitches, active_timesteps = np.where(piano_roll > 0)

    # Iterate through all possibile pitches in different time steps.
    for pitch, t in zip(active_pitches, active_timesteps):
        
       # Save original velocity.
        velocity = piano_roll[pitch, t]
    
        # Move the pitch in the range lb-up.
        # 12 => shift by one octave.
        while pitch < lb:
            pitch += 12
        while pitch > ub:
            pitch -= 12
        
        # Write the note in the piano roll.
        if lb <= pitch <= ub:
            normalized_piano_roll[pitch, t] = velocity

    return normalized_piano_roll

def show_piano_roll(piano_roll, slicing=None):
    """This function is used to print the piano roll, getting the piano roll object
    and eventually the interval (through the slicing).

    Arguments:
        piano_roll   This is simply the piano roll object it prints.
        slicing      This is the slicing of the columns and rows of the piano roll.
    """
    plt.figure(figsize=(12, 4))
    if slicing is not None:
        plt.imshow(piano_roll[slicing], aspect="auto", origin="lower", cmap="hot")
    else:
        plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="hot")
    plt.title("MIDI piano roll")
    plt.ylabel("Pitch (MIDI note)")
    plt.xlabel("Time (frames)")
    plt.colorbar(label="Velocity")
    plt.show()


def get_bar_duration(midi):
    """This function return the duration in seconds of the pretty midi object.

    Arguments:
        midi   The pretty midi object we need to get the duration of.
    """
    # Get tempo (assume constant tempo).
    bpm = midi.get_tempo_changes()[1][0]  # Beat Per Minute.

    # Get time signature (assumes constant time signature)
    time_signature = midi.time_signature_changes[0]
    numerator = time_signature.numerator  # Beats per bar
    denominator = time_signature.denominator  # Beat note type

    # Duration of one beat in seconds = 60 / BPM.
    beat_duration = 60.0 / bpm

    # Duration of one bar.
    bar_duration = beat_duration * numerator

    return bar_duration


# Given the midi file, returns a piano roll containing the melody.
# The melody is obtained by considering the note with the hightest pitch
# at each time step.
def extract_melody(midi, fs):
    """This function extract the melody (piano roll object with only one note per
    time steps).

    Arguments:
        midi   This is the input pretty midi object it work with.
        fs     This is the frame per seconds we read the midi file with.
    """
    # Get the original piano roll of 'midi'.
    full_roll = midi.get_piano_roll(fs=fs)

    # Initialize melody piano roll.
    melody_roll = np.zeros_like(full_roll)

    # Define the melody.
    for t in range(full_roll.shape[1]):

        # Consider notes at time step 't' (i.e. a column).
        active = full_roll[:, t]

        # If there exist a note with pitch != 0 ...
        if np.any(active):
            highest_pitch = np.argmax(active)
            melody_roll[highest_pitch, t] = active[highest_pitch]

    return melody_roll


def compute_variance(midi, attribute):
    """This function is used to compute the variance of the notes attribute (given in input).

    Attribute:
        midi        This is the pretty midi input object.
        attribute   This is the attribute we compute the variance of.
    """

    values = []

    # Iterate for each instrument.
    for instrument in midi.instruments:
        # Iterate for each note in the instrument.
        for n in instrument.notes:
            # Selecting the correct attribute (three possible) and store
            # values.
            if attribute == "velocity":
                values.append(n.velocity)

            if attribute == "duration":
                values.append(n.end - n.start)

            if attribute == "pitch":
                values.append(n.pitch)
    if not values:
        return 0.0
    # Compute and return variance of the list.
    return np.var(values)


def midi_selection(midi, fs) -> bool:
    """This function apply a simple criteria based on the amount of notes played in the same
    time steps along the midi (not the melody).

    Arguments:
        midi   This is the pretty_midi object.
        fs     This is the frame rate we read the pretty midi object with.
    """

    # Get the full piano roll from midi object.
    full_roll = midi.get_piano_roll(fs=fs)

    # Now we check how many notes are played.
    notes_per_time_frame = np.sum(full_roll > 0, axis=0)

    max_polyphony = np.max(notes_per_time_frame) if notes_per_time_frame.size > 0 else 0

    # Logic behind midi selection.
    if max_polyphony > MIN_PIANO_NOTES and max_polyphony < MAX_PIANO_NOTES:
        return True
    return False


def melody_selection(melody_roll):
    """This function apply a simple criteria based on the amount of notes played in the initial
    melody.
    Arguments:
            melody_roll   This is a piano_roll object.
    """

    notes, frames = melody_roll.shape

    unique_notes = set()

    # Iterate through all frames.
    for f in range(frames):
        # Iterate through all notes.
        for n in range(notes):
            # Check the presence of notes.
            if melody_roll[n, f] > 0:
                # Store unique notes.
                if n not in unique_notes:
                    unique_notes.add(n)
    if len(unique_notes) > 5 and len(unique_notes) < 15:
        return True
    return False


def chords_finder(piano_roll):
    """
    This function identify the chords (minor or major) present in the given piano roll column.
    Explanation:
        1) compress all the piano roll notes (all octaves, 12 semitones) into one octave (12
        semitones).
        2) for all possible chords (minor and major) check if the first note is present.
        3) if a note is present then if it is minor use the shifts [0,3,7] to check the presence
        of other notes, instead if it is miajor use the shifts [0,4,7].
        4) return a binary vector of size 13 (major or minor as 0 or 1 in last position) with
        ones in position of present chords.
    Args:
        piano_roll   This is a piano roll object.
    """

    # This is the vector of chords returned in the end.
    major_chords = np.zeros(13)
    major_chords[12] = 0
    minor_chords = np.zeros(13)
    minor_chords[12] = 1

    # Collapsing all the notes into one octave.
    collapsed_octave = np.zeros(12)
    for i in range(piano_roll.shape[0]):
        if piano_roll[i] > 0:
            collapsed_octave[i % 12] = 1

    # Checking presence of major chords (pattern [0,4,7]).
    for i in range(len(MAJOR_CHORDS)):
        if (
            collapsed_octave[MAJOR_CHORDS[i]] == 1
            and collapsed_octave[(MAJOR_CHORDS[i] + 4) % 12] == 1
            and collapsed_octave[(MAJOR_CHORDS[i] + 7) % 12] == 1
        ):
            major_chords[MAJOR_CHORDS[i]] = 1

    # Checking presence of minor chords (pattern [0,3,7]).
    for i in range(len(MINOR_CHORDS)):
        if (
            collapsed_octave[MINOR_CHORDS[i]] == 1
            and collapsed_octave[(MINOR_CHORDS[i] + 3) % 12] == 1
            and collapsed_octave[(MINOR_CHORDS[i] + 7) % 12] == 1
        ):
            minor_chords[MINOR_CHORDS[i]] = 1

    return minor_chords, major_chords


def main_chords(bar):
    """This function is used to extract the main chord among all possible chords present
    in the bar in different time steps.

    Arguments:
        bar   This a piano roll object with size of a single bar.
    """

    # Current chords detected in current bar.
    minor_curr = np.zeros(12)
    major_curr = np.zeros(12)

    # Here we count the amount of times we saw a the note in the
    # position.
    minor_chords_count = np.zeros(12)
    major_chords_count = np.zeros(12)

    # Final np.array with main chord.
    final_chords = np.zeros(13)

    max_freq = 0

    # Iterating through all time steps in the bar.
    for i in range(bar.shape[1]):
        minor_curr, major_curr = chords_finder(bar[:, i])
        minor_curr.resize(12)
        major_curr.resize(12)
        # Incrementing the counter corresponding to the chord we found.
        minor_chords_count = minor_chords_count + minor_curr
        major_chords_count = major_chords_count + major_curr

    # Getting the max frequency chord for both major and minor chords.
    max_freq_major = np.max(major_chords_count)
    max_freq_minor = np.max(minor_chords_count)

    # Getting the index of best chord.
    if max_freq_major > max_freq_minor:
        max_pos = np.argmax(major_chords_count)
        final_chords[12] = 0
    else:
        final_chords[12] = 1
        max_pos = np.argmax(minor_chords_count)

    # Returning the main chords selected based on frequency.
    final_chords[max_pos] = 1
    return final_chords


# Construct the pretty midi object starting from the given piano roll.
# Returns the pretty midi object.
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    """This function has the purpose to return the reconstructed pretty midi
    from the piano roll.

    Arguments:
        piano_roll   The piano roll object in input.
        fs           This is the frame we get per second from the piano roll.
        program      This is the instrument we use for the pretty midi object.
    """

    # Getting the dimensions of the piano roll.
    notes, frames = piano_roll.shape

    # Creating the piano roll object.
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Pad 1 column of zeros so we can acknowledge initial and ending events.
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")

    # Use changes in velocities to find note on / note off events.
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # Keep track of note on times.
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    # Time is frame number, note is pitch.
    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    pm.instruments.append(instrument)
    return pm


def midi_splitter(midi, out_list, fs=8, n_bars=8):
    """This function split pretty midi object in elements with a given
    amount of bars.

    Arguments:
        midi       This is the input pretty midi object it divides.
        out_list   This is the output object which is a list of midi splits.
        fs         This is the frame per second we read the midi with.
        n_bars     This is the amount of bars each split will have.
    """

    # Get piano roll.
    midi_roll = midi.get_piano_roll(fs=fs)

    # Total number of splits.
    splits = int(midi_roll.shape[1] / (COLS_PER_BAR * n_bars))

    # Total number of columns per split.
    n_cols = COLS_PER_BAR * n_bars

    # Split into samples of size 128xCOLS_PER_BAR.
    for i in range(splits):
        tmp = i * n_cols
        sample = midi_roll[:, tmp : tmp + n_cols]

        # Convert to midi file and store sample.
        sample = piano_roll_to_pretty_midi(sample, fs)
        out_list.append(sample)


def check_octave(bar):
    """This function is used to check whether a bar has one shifted note
    out of the admissible range for both the 2 octaves considered.

    Arguments:
        bar   This is a piano roll object.
    """
    # If the exceeded note of the first octave is not empty we fix its position.
    if not (np.array_equal(bar[73, :], EMPTY)):
        bar[60, :] = bar[73, :]
        bar[73, :] = EMPTY

    # If the exceeded note of the second octave is not empty we fix its position.
    if not (np.array_equal(bar[84, :], EMPTY)):
        bar[73, :] = bar[84, :]
        bar[84, :] = EMPTY


def shift_roll_up(sample, n):
    """This function can be used to circularly shift by n semitones the 2 bars
    checking for their integrity after the shift.

    Arguments:
        sample   This is a sample with previous and current bar  (pair object).
        n        This is the amount of semitones the notes in sample need to shift of.
    """

    # Getting the 2 bar contained in a sample (previous and current).
    prev, curr = sample

    # Iterating through all values in the shift range.
    for i in range(n):
        # Adding one fake note at the beginning of the roll.
        prev = np.vstack((EMPTY, prev))
        # Copying last note in the first one.
        prev[0, :] = prev[128, :]
        # Deleting the last note (we have an excess of notes by one).
        prev = np.delete(prev, 128, 0)

        # Adding one fake note at the beginning of the roll.
        curr = np.vstack((EMPTY, curr))
        # Copying last note in the first one.
        curr[0, :] = curr[128, :]
        # Deleting the last note (we have an excess of notes by one).
        curr = np.delete(curr, 128, 0)

        # Check on prev and curr for both the last note of the 2 considered octaves.
        check_octave(prev)
        check_octave(curr)

    # Transform in bool.
    prev = prev.astype(bool)
    curr = curr.astype(bool)

    return prev, curr


def get_sample_pairs(midi, out_list, fs=8):
    """This function is used to get the sample pairs from a pretty midi
    object (previous bar, current bar).

    Arguments:
        midi       This is the input pretty midi object.
        out_list   This a list we put all sample pairs found.
        fs         This is the frame per second we read the midi file with.
    """

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
        melody_roll, ((0, 0), (0, zero_padding)), mode="constant", constant_values=0
    )

    # Fill piano roll, i.e. remove pauses.
    melody_roll = fill_melody_pauses(melody_roll)

    # Split into samples of size 128xCOLS_PER_BAR matrices.
    splitted = []
    splits = int(melody_roll.shape[1] / COLS_PER_BAR)
    for i in range(splits):
        tmp = i * COLS_PER_BAR
        sample = melody_roll[:, tmp : tmp + COLS_PER_BAR]
        splitted.append(sample)

    # Create the pairs of previous and current bars.
    for i in range(1, len(splitted)):
        pair = splitted[i - 1], splitted[i]
        out_list.append(pair)


def get_sample_triplets(midi, out_list, labels, fs=8):
    """This function is used to get the sample triplets from a pretty midi
    object (previous bar, current bar, previous chord).

    Arguments:
        midi       This is the input pretty midi object.
        out_list   This a list we put all sample triplets found.
        fs         This is the frame per second we read the midi file with.
    """

    # Create meoldy piano roll.
    melody_roll = extract_melody(midi, fs=fs)
    # Get the full piano roll from the midi
    piano_roll = midi.get_piano_roll(fs=fs)

    # Binarize melody piano roll.
    melody_roll[melody_roll > 0] = 1
    melody_roll = melody_roll.astype(bool)

    # Binarize full piano roll
    piano_roll[piano_roll > 0] = 1
    piano_roll = piano_roll.astype(bool)

    # Normalize melody piano roll.
    melody_roll = normalize_melody_roll(melody_roll, lb=60, ub=83)

    # Add zero padding to the end of the piano rolls (if necessary).
    zero_padding = melody_roll.shape[1] % COLS_PER_BAR
    melody_roll = np.pad(
        melody_roll, ((0, 0), (0, zero_padding)), mode="constant", constant_values=0
    )
    piano_roll = np.pad(
        piano_roll, ((0, 0), (0, zero_padding)), mode="constant", constant_values=0
    )

    # Fill piano roll, i.e. remove pauses.
    melody_roll = fill_melody_pauses(melody_roll)

    # Split into samples of size 128xCOLS_PER_BAR matrices.
    splitted = []
    splitted_full = []
    splits = int(melody_roll.shape[1] / COLS_PER_BAR)
    for i in range(splits):
        tmp = i * COLS_PER_BAR
        sample = melody_roll[:, tmp : tmp + COLS_PER_BAR]
        full_sample = piano_roll[:, tmp : tmp + COLS_PER_BAR]
        splitted.append(sample)
        splitted_full.append(full_sample)

    # Create the pairs of previous and current bars.
    for i in range(1, len(splitted)):
        main_chord = main_chords(splitted_full[i]).astype(bool)
        pair = splitted[i - 1], splitted[i]
        labels.append(main_chord)
        out_list.append(pair)


def get_midi_from_dir(dir_path):
    """This function returns all the pretty midi objects inside the dir_path.

    Arguments:
        dir_path   This is a string with the path of the input directory.
    """

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
