import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

MAX_PIANO_NOTES = 20
MIN_PIANO_NOTES = 5

def fill_melody_pauses(piano_roll):
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
            filled_roll[first_active_pitch, t] = filled_roll[first_active_pitch, first_active_t]

    return filled_roll

def normalize_melody_roll(piano_roll, lb, ub):
    # Output roll.
    normalized_piano_roll = np.zeros_like(piano_roll)
    
    # Finds coordinates of active notes.
    active_pitches, active_timesteps = np.where(piano_roll > 0)
    
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
    plt.figure(figsize=(12, 4))
    if slicing is not None:
        plt.imshow(piano_roll[slicing], aspect='auto', origin='lower', cmap='hot')
    else:
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot')
    plt.title('MIDI piano roll')
    plt.ylabel('Pitch (MIDI note)')
    plt.xlabel('Time (frames)')
    plt.colorbar(label='Velocity')
    plt.show()

# Given the midi file, returns the bar duration in secons.
def get_bar_duration(midi):
    # Get tempo (assume constant tempo).
    bpm = midi.get_tempo_changes()[1][0]  # Beat Per Minute.
    
    # Get time signature (assumes constant time signature)
    time_signature = midi.time_signature_changes[0]
    numerator = time_signature.numerator      # Beats per bar
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


def compute_variance(midi, attribute) -> float:
    values = []

    
    for instrument in midi.instruments:
        for n in instrument.notes:
            if attribute == "velocity":
                values.append(n.velocity)
            
            if attribute == "duration":
                values.append(n.end - n.start)
            
            if attribute == "pitch":
                values.append(n.pitch)
    if not values: 
        return 0.0
        
    return np.var(values)
    
def midi_selection(midi, fs) -> bool:
    """
        This function apply a simple criteria based on the amount of notes played in the same instant along  
        the midi (not the melody)
        Args: 
            -midi: pretty_midi
            -fs:   int
    """
    #Get the full piano roll from midi object
    full_roll = midi.get_piano_roll(fs = fs)

    # Now we check how many notes are played 
    notes_per_time_frame = np.sum(full_roll > 0, axis=0)

    max_polyphony = np.max(notes_per_time_frame) if notes_per_time_frame.size > 0 else 0
    
    # Logic behind midi selection
    if max_polyphony > MIN_PIANO_NOTES and max_polyphony < MAX_PIANO_NOTES:
        return True
    return False

def melody_selection(melody_roll):
    """
        This function apply a simple criteria based on the amount of notes played in the initial 
        melody
        Args: 
            -melody_roll: piano_roll
    """
    notes, frames = melody_roll.shape

    unique_notes = set()
    
    for f in range(frames):
        for n in range(notes):
            if melody_roll[n, f] > 0:
                if n not in unique_notes:
                    unique_notes.add(n)
    if len(unique_notes) > 5 and len(unique_notes) < 15:
        return True
    return False

def octave_sum(melody_roll):
    notes, frames = melody_roll.shape
    for f in range(frames):
        for n in range(notes-1):
            melody_roll[n + 1,f] = melody_roll[n,f]
        melody_roll[0,f] = melody_roll[notes - 1, f] 
        
    melody_roll = normalize_melody_roll(melody_roll, lb=60, ub=83)
    
    return melody_roll
    

# Construct the pretty midi object starting from the given piano roll.
# Returns the pretty midi object.
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Pad 1 column of zeros so we can acknowledge initial and ending events.
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # Use changes in velocities to find note on / note off events.
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # Keep track of note on times.
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # time is frame number, note is pitch
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
                end=time
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    pm.instruments.append(instrument)
    return pm
