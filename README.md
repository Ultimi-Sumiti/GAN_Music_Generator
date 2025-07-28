# Symbolic-Domain Music Generation with GANs

![Music Generation](https://img.shields.io/badge/domain-music%20generation-blue) 
![GANs](https://img.shields.io/badge/model-GANs-orange) 
![MIDI](https://img.shields.io/badge/data-MIDI-lightgrey)
# Abstract
*"Recent advances in generative models have made the automated production of music a timely and important area of deep learning research.This project presents a simplified and effective Generative Adversarial Network (GAN), inspired by [MidiNet paper](https://arxiv.org/abs/1703.10847), for symbolic music generation. Its importance lies in demonstrating that a minimal, interpretable model can achieve stable and musically coherent results by addressing practical training challenges like mode collapse and non-convergence. This is accomplished through techniques like minibatch discrimination and careful hyperparameter tuning.The main result is a successfully balanced training process, enabling the generator to produce structured piano roll melodies without collapsing. This work provides a reproducible and efficient baseline that can serve as a practical foundation for further experimental research in music generation."*

# How It Works

The main part of this project is a GAN architecture designed for creating melodies. The entire process, from raw data to generated music, follows a specific pipeline:

1.  **Input Data**: The process starts with raw MIDI files from the MAESTRO Dataset.
2.  **Preprocessing**: Each MIDI file is converted into a piano roll representation, a binary matrix where notes are mapped over time steps. This stage includes extracting the main melody from chords and removing pauses.
3.  **Data Augmentation**: To increase the dataset size, the melodies are circularly shifted up by one semitone recursively 11 times, creating 12 versions of each melody bar.
4.  **GAN Training**: The augmented dataset is used to train the GAN models. The training treats common GAN issues like mode collapse and instability through fine-tuned hyperparameters and techniques like mini-batch discrimination.
5.  **Music Generation**: Once trained, the generator model can create new piano roll sequences, which are then converted back into MIDI files that can be stored and replayed.


# Models

This project implements three distinct GAN models with increasing complexity:

* **`model_v1`**: A baseline DCGAN that generates single, one-bar-long melodies from a random noise vector.
* **`model_v2`**: A Conditional DCGAN that generates a melody bar conditioned on the preceding bar, encouraging more harmonically coherent sequences.
* **`model_v3`**: An extension of `model_v2` that is also conditioned on the chord associated with the previous bar, adding another layer of musical context to the generation process.

***

# Key Musical Concepts
This section briefly defines the principal musical terms present in this project.

**Piano Roll**: A binary matrix representing musical data. It maps 128 MIDI notes in 16 time-steps per bar.

**Melody**: A monophonic sequence where only one note is active at each time-step. It is extracted by selecting the highest-velocity note from the full piano roll in each frame.

**Chord**: A 13-dimensional vector that encodes the chord iteself, specifying the first key and whether the chord is major or minor. It is derived from the most frequent chord in the previous bar of music.

**Octave**: All musical notes are normalized into a fixed two-octave range (MIDI notes 60-83 equivalent to C4-B5). This helps in detecting training issues like mode collapse.

# Dataset used:
- **[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)**: 200+ hours of piano performances.
We mainly used the MAESTRO Dataset. In this repository you can find the version 3.0.0.

---
## Project structure
```bash
.
├── data
│   ├── preprocessed
│   │   └── maestro-v3.0.0
│   │       ├── dataset1          # Datasets for model_v1      
│   │       ├── dataset2          # Datasets for model_v2
│   │       └── dataset3          # Datasets for model_v3
│   └── raw
│       └── maestro-v3.0.0        # The Maestro Dataset
│
├── models                        # Contains the source code of model_v1, model_v2, model_v3
├── outputs
│   ├── checkpoints               # Checkpoints of the three models, divided by dataset
│   └── songs                     # Some good songs ouputted by the three models
│
├── utils                         # Contains functions that are used in different parts of the project
│
├── tester_model_v2.ipynb         # Notebook used to test a trained model_v2
├── tester_model_v3.ipynb         # Notebook used to test a trained model_v3
├── train_model_v1.ipynb          # Notebook used to train a model_v1
├── train_model_v2.ipynb          # Notebook used to train a model_v2
└── train_model_v3.ipynb          # Notebook used to train a model_v3
```

---
##  Setup
1. Clone the repo:  
   ```bash
   git clone https://github.com/Ultimi-Sumiti/DL_project/GAN_Music_Generator.git

2. Install dependencies:
   ```bash  
   pip install -r requirements.txt
---

## Team Members
- **Luca Piai**   [GitHub](https://github.com/luca037)   

- **Alessandro chinello** [GitHub](https://github.com/Ale10chine) 

- **Mattia Scantamburlo**  [GitHub](https://github.com/Daedalus02)  


 






