import numpy as np

# Major chords first notes (starts from C = 1th note)
MAJOR_CHORDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Minor chords first notes (starts from A = 10th note)
MINOR_CHORDS = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def chords_finder(piano_roll):
    """
        This function identify the chords (minor or major) present in the given piano roll column.
        Explanation:
            1) compress all the piano roll notes (all octaves, 12 semitones) into one octave (12 semitones)
            2) for all possible chords (minor and major) check if the first note is present
            3) if a note is present then if it is minor use the shifts [0,3,7] to check the presence
                of other notes, instead if it is miajor use the shifts [0,4,7]
            4) return a binary vector of size 13 (major or minor as 0 or 1 in last position) with ones 
                in position of present chords
        Args: 
            -piano_roll: piano_roll
    """
   # if piano_roll.shape[1] != 1:
   #     return

    # This is the vector of chords returned in the end
    major_chords = np.zeros(13)
    major_chords[12] = 0
    minor_chords = np.zeros(13)
    minor_chords[12] = 1
    
    # Collapsing all the notes into one octave 
    collapsed_octave = np.zeros(12)
    for i in range(piano_roll.shape[0]):
        if piano_roll[i] > 0:
            collapsed_octave[i%12]  = 1

    # Checking presence of major chords (pattern [0,4,7])
    for i in range(len(MAJOR_CHORDS)):
        if collapsed_octave[MAJOR_CHORDS[i]] == 1 and collapsed_octave[(MAJOR_CHORDS[i] + 4)%12] == 1 and collapsed_octave[(MAJOR_CHORDS[i] + 7)%12] == 1:
            major_chords[MAJOR_CHORDS[i]] = 1

    # Checking presence of minor chords (pattern [0,3,7])
    for i in range(len(MINOR_CHORDS)):
        if collapsed_octave[MINOR_CHORDS[i]] == 1 and collapsed_octave[(MINOR_CHORDS[i] + 3)%12] == 1 and collapsed_octave[(MINOR_CHORDS[i] + 7)%12] == 1:
            minor_chords[MINOR_CHORDS[i]] = 1
    
    return minor_chords, major_chords



def main_chords (bar):
    minor_curr = np.zeros(12)
    major_curr = np.zeros(12)
    
    minor_chords_count = np.zeros(12)
    major_chords_count = np.zeros(12)
    
    main_chord_major = np.zeros(12)
    main_chord_minor = np.zeros(12)
    final_chords = np.zeros(13)
    
    max_freq = 0
    
    for i in range(bar.shape[1]):
        minor_curr, major_curr = chords_finder(bar[:,i])
        minor_curr.resize(12)
        major_curr.resize(12)
        minor_chords_count = minor_chords_count + minor_curr
        major_chords_count = major_chords_count + major_curr
        

    max_freq_major = np.max(major_chords_count)
    max_freq_minor = np.max(minor_chords_count)
    if max_freq_major > max_freq_minor:
        max_pos = np.argmax(major_chords_count)
        final_chords[12] = 0
    else:
        final_chords[12] = 1
        max_pos = np.argmax(minor_chords_count)
        
    final_chords[max_pos] = 1
    return final_chords
    

