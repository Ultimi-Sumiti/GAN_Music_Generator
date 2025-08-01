# Symbolic-Domain Music Generation with GANs

![Music Generation](https://img.shields.io/badge/domain-music%20generation-blue) 
![GANs](https://img.shields.io/badge/model-GANs-orange) 
![MIDI](https://img.shields.io/badge/data-MIDI-lightgrey)

https://github.com/user-attachments/assets/82d527f9-1e5f-466a-a88c-6aa98c1c3a07

## Table of Contents
- [Abstract](#abstract)
- [Key Musical Concepts](#key-musical-concepts)
- [How It Works](#how-it-works)
- [Dataset used](#dataset)
  - [How to create a dataset starting from raw MIDI files](#how-to-create-a-dataset-starting-from-raw-midi-files)
- [Models](#models)
  - [How to train a model](#how-to-train-a-model)
  - [How to test a model](#how-to-test-a-model)
- [Project structure](#project-structure)
- [Download project](#download-project)
- [Results](#results)
  - [Songs](#songs)

# Abstract 
*"Recent advances in generative models have made the automated production of music an important area of deep learning research. This paper presents a simplified Generative Adversarial Network (GAN), inspired by [MidiNet paper](https://arxiv.org/abs/1703.10847), for symbolic music generation using the MAESTRO dataset. Its importance is in the fact that it demonstrates that a minimal, interpretable model can achieve stable and musically coherent results by addressing practical training challenges like mode collapse and non-convergence using techniques like minibatch discrimination and hyperparameter tuning. The main result is a successful training process, achieved by adjusting learning rates and update steps. This enables the generator to produce piano roll melodies without collapsing. This work provides a reproducible baseline that can be used as a good practical starting point for other experimental research in music generation."*

More details can be found in `Symbolic-Domain_Music_Generation_with_GANs.pdf`.


# Key Musical Concepts
This section briefly defines the principal musical terms present in this project.

**Piano Roll**: A binary matrix representing musical data. It maps 128 MIDI notes in 16 time-steps per bar.

**Melody**: A monophonic sequence where only one note is active at each time-step. It is extracted by selecting the highest-velocity note from the full piano roll in each frame.

**Chord**: A 13-dimensional vector that encodes the chord iteself, specifying the first key and whether the chord is major or minor. It is derived from the most frequent chord in the previous bar of music.

**Octave**: All musical notes are normalized into a fixed two-octave range (MIDI notes 60-83 equivalent to C4-B5). This helps in detecting training issues like mode collapse.


# How It Works
The main part of this project is a GAN architecture designed for creating melodies. The entire process, from raw data to generated music, follows a specific pipeline:

1.  **Input Data**: The process starts with raw MIDI files from the MAESTRO Dataset.
2.  **Preprocessing**: Each MIDI file is converted into a piano roll representation, a binary matrix where notes are mapped over time steps. This stage includes extracting the main melody from chords and removing pauses.
3.  **Data Augmentation**: To increase the dataset size, the melodies are circularly shifted up by one semitone recursively 11 times, creating 12 versions of each melody bar.
4.  **GAN Training**: The augmented dataset is used to train the GAN models. The training treats common GAN issues like mode collapse and instability through fine-tuned hyperparameters and techniques like mini-batch discrimination.
5.  **Music Generation**: Once trained, the generator model can create new piano roll sequences, which are then converted back into MIDI files that can be stored and replayed.

---

# Dataset
- **[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)**: 200+ hours of piano performances.

We also used **[The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)**
to perform some tests, but here we have not reported a snapshot of this dataset.

## How to create a dataset starting from raw MIDI files
Consider that we want create a dataset for `model_v2`.
You need to open the file `create_dataset2.py` in `data` folder and modfy the global variables:
```python
# Define the directory of the input midi file.
# The specified directory must contains '.midi' and/or '.mid' files.
INPUT_DIR_PATH = "./raw/maestro-v3.0.0/all/"

# Define where the dataset will be saved.
OUT_DIR_PATH = "./preprocessed/maestro-v3.0.0/dataset2/"

# Define the name of the dataset (must end with '.h5').
OUT_FILE_NAME = "all.h5"
```
Note that by defalut the name of input directory path is `./raw/maestro-v3.0.0/all/` but this directory is not provided.
During our study, we manually copied the entire MAESTRO dataset into this directory.
This setup allowed us to randomly select files from the dataset to create a testing dataset.

Finaly you can create the dataset with:
```bash
python create_dataset2.py
```
A compressed file will be crated in the specified path.

By default we perform data augmentation when creating the dataset.
In general this is not a good practice, but in this way we were able to pre-load the entire dataset
entirely in GPU before starting the training of the architecture.
The dataset contains binary matrices, so it's size should be relatively small and shuld fit entirely on the GPU.

# Models
This project implements three distinct GAN models with increasing complexity:

* **`model_v1`**: A baseline DCGAN that generates single, one-bar-long melodies from a random noise vector.
* **`model_v2`**: A Conditional DCGAN that generates a melody bar conditioned on the preceding bar, encouraging more harmonically coherent sequences.
* **`model_v3`**: An extension of `model_v2` that is also conditioned on the chord associated with the previous bar, adding another layer of musical context to the generation process.
  
## How to train a model
Consider that we want to train `model_v2` using the dataset stored in `/preprocessed/maestro-v3.0.0/dataset2/dataset_name.h5`.
We suggest to use `model_v2` or `model_v3`, since the first model generates very short melodies that are not particularly
interesting to listen to.

Open notebook `train_model_v2.ipynb`, go to the "Setup" section and change the value of the `DATASET_PATH` variable.
```python
# Dataset path.
DATASET_PATH = "data/preprocessed/maestro-v3.0.0/dataset2/dataset_name.h5"
```
The just run all the cells to train the model.

## How to test a model
To test a model you should open notebook `tester_model_v2.ipynb` and change the line that load the checkpiont.
```python
# Load model from checkpoint.
ckp_path = "checkpoint_path.ckpt"
model = GAN.load_from_checkpoint(ckp_path)
```
Then you can just run the notebook, 10 MIDI files will be created in `./outputs/songs/`.

If instead you want to generate melodies using `model_v3` you need to open notebook `tester_model_v3.ipynb`.
You need to perform the same steps described above.
At the end of this file you can see that, for a single melody, 6 MIDI files are created.
Each of them differs from the "chord pattern" applied.

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
│   ├── raw
│   │   └── maestro-v3.0.0        # The Maestro Dataset
│   │
│   ├── create_dataset1.py        # Used to create a dataset for model_v1
│   ├── create_dataset2.py        # Used to create a dataset for model_v2
│   ├── create_dataset3.py        # Used to create a dataset for model_v3
│   │
│   └── midi_preprocessing.py     # Contains all the functions used to create the dataset.
│
├── models                        # Contains the source code of model_v1, model_v2, model_v3
│
├── outputs
│   ├── checkpoints               # Checkpoints of the three models, divided by dataset
│   └── songs                     # Some good songs ouputted by the three models
│
├── utils                         # Contains utils functions  used in different parts of the project
│
├── tester_model_v2.ipynb         # Notebook used to test a trained model_v2
├── tester_model_v3.ipynb         # Notebook used to test a trained model_v3
├── train_model_v1.ipynb          # Notebook used to train a model_v1
├── train_model_v2.ipynb          # Notebook used to train a model_v2
└── train_model_v3.ipynb          # Notebook used to train a model_v3
```

---
##  Download project
1. Clone the repo:  
   ```bash
   git clone https://github.com/Ultimi-Sumiti/DL_project/GAN_Music_Generator.git

2. Install dependencies:
   ```bash  
   pip install -r requirements.txt
---

## Results
*(Note that the reported songs were not generated only with the models that you can load using the checkpoints.)*

## Songs
In the following we report the ouputs inside the `songs` directory so you can listen to them direclty here without downloading the files.
Also, in the video, you can visualize the piano roll of the melody.

We do not provide samples generated my `model_v1` since they are not particularly interesting to listen to.

Melodies generated after training on the MAESTRO dataset have filenames that start with "maestro".
Some melodies were generated after training on the full ABBA directory from the Lakh MIDI Dataset; these can be recognized by filenames starting with "abba".

Note that some melodies generated by `model_v2` are 9 bars long.
This was due to a mistake in the code.

### Model v2
Only melody, no chords.

https://github.com/user-attachments/assets/d522a59a-8539-4017-bd6e-0a13f38e957a

https://github.com/user-attachments/assets/eb8112cd-4776-4ab6-88a3-96214156483b

https://github.com/user-attachments/assets/c112088c-17a2-4a39-b069-adec21f771b7

https://github.com/user-attachments/assets/4f056c07-497a-4525-a695-9d3e275cf675

https://github.com/user-attachments/assets/99fbfd79-9cb1-41a1-981a-8f13c762fcc9

https://github.com/user-attachments/assets/eef2ab64-3aa4-41c5-91ab-95d18cd52a21

https://github.com/user-attachments/assets/ff616048-7707-43b6-a443-ae5c6c5c70bd

https://github.com/user-attachments/assets/77605500-b42c-44e7-a4d4-3063df87cdf0

### Model v3
Melody and chords.

Melodies generated with this model have lower quality compared to those produced by the previous version.
This is partly because we dedicated significantly more time to tuning and experimenting with Model v2.
We observed that training Model v3 was more challenging: finding suitable 
hyperparameter values to stabilize the $\min\max$ game was more difficult than with Model v2.

https://github.com/user-attachments/assets/c328bcd5-630b-4c0b-a7ff-685774bf2356

https://github.com/user-attachments/assets/a89f8a69-ad99-46c5-a8c9-ac113ff7b18b

https://github.com/user-attachments/assets/304b05e2-6a8d-4875-8d66-d21cb8c45263
