# Symbolic-Domain Music Generation with GANs

![Music Generation](https://img.shields.io/badge/domain-music%20generation-blue) 
![GANs](https://img.shields.io/badge/model-GANs-orange) 
![MIDI](https://img.shields.io/badge/data-MIDI-lightgrey)
# Abstract
*Recent advances in generative models have made the automated production of music a timely and important area of deep learning research.This project presents a simplified and effective Generative Adversarial Network (GAN), inspired by MidiNet, for symbolic music generation. Its importance lies in demonstrating that a minimal, interpretable model can achieve stable and musically coherent results by addressing practical training challenges like mode collapse and non-convergence. This is accomplished through techniques like minibatch discrimination and careful hyperparameter tuning.The main result is a successfully balanced training process, enabling the generator to produce structured piano roll melodies without collapsing. This work provides a reproducible and efficient baseline that can serve as a practical foundation for further experimental research in music generation.*

Music generation involves creating musical pieces using neural networks. This project focuses on **symbolic representations** (e.g., MIDI files), which encode notes, instruments, tempo, and other metadata, rather than raw audio.

In particular in this project we have explored symbolic-domain music generation using Generative Adversarial Networks (GANs), inspired by the [MidiNet paper](https://arxiv.org/abs/1703.10847). The goal is to generate melodies or full musical pieces from MIDI data. 
We have initially try to use a very basic implementation inspired by the simplest model in the paper, removing the conditioner network and any kind of conditioning method. After that we improved the first implementation with 2 type of conditioning methods tackled by the paper.


Project overview

Project structure

Task tackled

Ricordare i termini basici per la musica come melodia chord etc

Dataset used:
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


 






