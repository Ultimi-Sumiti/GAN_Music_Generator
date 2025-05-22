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
├── 📁 data/ # Raw/processed data (excluded from version control)
│ ├── raw/ # Original datasets (e.g., MIDI files)
│ └── processed/ # Preprocessed data for training
│
├── 📁 models/ # Model architectures
│ ├── generator.py # GAN generator
│ └── discriminator.py # GAN discriminator
│
├── 📁 utils/ # Utilities
│ ├── dataset_loader.py # MIDI data loading
│ ├── metrics.py # Evaluation metrics
│ └── plot_tools.py # Visualization tools
│
├── 📁 configs/ # Training configurations (YAML/JSON)
│ └── default.yaml # Hyperparameters
│
├── 📁 training/ # Training scripts
│ ├── train.py # Main training loop
│ └── trainer.py # Training logic
│
├── 📁 evaluation/ # Model evaluation
│ └── evaluate.py # Metrics/comparisons
│
├── 📁 experiments/ # Experiment logs/notebooks
│ └── exp1_gan_vs_wgan.ipynb # Example experiment
│
├── 📁 outputs/ # Generated outputs
│ ├── images/ # Sample outputs
│ ├── checkpoints/ # Saved models
│ └── logs/ # Training logs
│
├── .gitignore # Excluded files (e.g., large datasets)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── main.py # Entry point for training



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

   
