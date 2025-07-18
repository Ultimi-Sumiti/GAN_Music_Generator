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
        prev = bars[i-1]
    
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
