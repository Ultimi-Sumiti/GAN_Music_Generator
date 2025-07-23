# Symbolic-Domain Music Generation with GANs

![Music Generation](https://img.shields.io/badge/domain-music%20generation-blue) 
![GANs](https://img.shields.io/badge/model-GANs-orange) 
![MIDI](https://img.shields.io/badge/data-MIDI-lightgrey)

Music generation involves creating musical pieces using neural networks. This project focuses on **symbolic representations** (e.g., MIDI files), which encode notes, instruments, tempo, and other metadata, rather than raw audio.

In particular in this project we have explored symbolic-domain music generation using Generative Adversarial Networks (GANs), inspired by the [MidiNet paper](https://arxiv.org/abs/1703.10847). The goal is to generate melodies or full musical pieces from MIDI data. 
We have initially try to use a very basic implementation inspired by the simplest model in the paper, removing the conditioner network and any kind of conditioning method. After that we improved the first implementation with 2 type of conditioning methods tackled by the paper.



Project overview

Project structure

Task tackled

Ricordare i termini basici per la musica come melodia chord etc

Dataset used
- **[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)**: 176K+ MIDI files (use the *Clean-MIDI subset*).  
- **[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)**: 200+ hours of piano performances.  

---

## 🔧 Setup
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


 






