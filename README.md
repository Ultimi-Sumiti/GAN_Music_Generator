# Symbolic-Domain Music Generation with GANs

![Music Generation](https://img.shields.io/badge/domain-music%20generation-blue) 
![GANs](https://img.shields.io/badge/model-GANs-orange) 
![MIDI](https://img.shields.io/badge/data-MIDI-lightgrey)

This project explores symbolic-domain music generation using Generative Adversarial Networks (GANs), inspired by the [MidiNet paper](https://arxiv.org/abs/1703.10847). The goal is to generate melodies or full musical pieces from MIDI data.

---

## 👥 Team Members
- **Luca Piai**   [GitHub](https://github.com/luca037)   

- **Alessandro chinello** [GitHub](https://github.com/Ale10chine) 

- **Mattia Scantamburlo**  [GitHub](https://github.com/Daedalus02)  

---

## 📌 Project Overview
### Symbolic-Domain Music Generation
Music generation involves creating musical pieces using neural networks. This project focuses on **symbolic representations** (e.g., MIDI files), which encode notes, instruments, tempo, and other metadata, rather than raw audio.

### MIDI Files
- **Structure**: Split into channels (one per instrument), each containing a *piano roll* (ordered notes with duration/velocity).  
- **Piano Roll**: Represented as an `M × N` matrix, where:  
  - `M` = Number of notes (low to high pitches).  
  - `N` = Number of timesteps.  

---

## 📂 Project Structure

gan-project/
│
├── 📁 data/                  # Raw or preprocessed data (do not version large files)
│   ├── raw/                 # Original data
│   └── processed/           # Preprocessed data for training
│
├── 📁 models/                # Network definitions (Generator, Discriminator, etc.)
│   ├── generator.py
│   └── discriminator.py
│
├── 📁 utils/                 # Various utilities (visualizations, metrics, helpers, etc.)
│   ├── dataset_loader.py
│   ├── metrics.py
│   └── plot_tools.py
│
├── 📁 configs/               # Configurations for training/testing (YAML or JSON)
│   └── default.yaml
│
├── 📁 training/              # Training scripts and logic
│   ├── train.py
│   └── trainer.py
│
├── 📁 evaluation/            # Scripts for model evaluation
│   └── evaluate.py
│
├── 📁 experiments/           # Notebooks, reports, or experiment logs
│   └── exp1_gan_vs_wgan.ipynb
│
├── 📁 outputs/               # Generated outputs (images, logs, saved models)
│   ├── images/
│   ├── checkpoints/
│   └── logs/
│
├── .gitignore               # Files to ignore (e.g., *.pt, __pycache__, data/raw, etc.)
├── requirements.txt         # Required Python libraries
├── README.md                # Main project documentation
└── main.py                  # Entry point for training or other tasks


---

## 🎵 Datasets
Suggested MIDI datasets:  
- **[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)**: 176K+ MIDI files (use the *Clean-MIDI subset*).  
- **[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)**: 200+ hours of piano performances.  

---

## 🎯 Tasks (Ordered by Difficulty)
1. **Melody generation** (single-note sequences).  
2. **Melody generation conditioned on previous notes**.  
3. **Melody generation conditioned on chords**.  
4. **Full musical piece generation** (melody + chords).  

---

## 💡 Suggestions
- **Preprocessing**: Spend time exploring MIDI data (e.g., isolating melodies/chords).  
- **Architectures**: Experiment with CNNs (e.g., ResNet, Inception) or VAEs.  
- **Music Theory**: Basic knowledge helps interpret data/literature.  

---

## 🎼 Musical Terminology
- **Melody**: Sequence of single notes over time.  
- **Chords**: Multiple notes played simultaneously.  
- **Harmony**: Chord progression underlying a melody.  

Example:  
- **Chords**: `C Major → G Major → A Minor`.  
- **Melody**: `C → E → G → A`.  

---

## 🔧 Setup
1. Clone the repo:  
   ```bash
   git clone https://github.com/Ultimi-Sumiti/DL_project/gan-project.git

2. Install dependencies:
   ```bash  
   pip install -r requirements.txt

   
