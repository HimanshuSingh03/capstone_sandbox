# Look Once to Hear - Complete Setup Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation & Setup](#installation--setup)
5. [Data Requirements](#data-requirements)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Model Architecture](#model-architecture)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Look Once to Hear** is an intelligent hearable system that enables users to selectively listen to a target speaker by simply looking at them for a few seconds. This system won the **Best Paper Honorable Mention 🏆 at CHI 2024**.

### How It Works

1. **"Look Once" (Enrollment Phase)**: 
   - User looks at target speaker for 3-5 seconds
   - System captures clean binaural audio of target speaker
   - Extracts speaker embedding (voice fingerprint)

2. **"To Hear" (Extraction Phase)**:
   - In noisy environment with multiple speakers
   - System identifies target speaker using stored embedding
   - Separates and enhances only the target speaker's voice
   - Outputs clean binaural audio of target speaker

### Key Technologies
- **Target Speaker Extraction**: Neural speech separation with speaker conditioning
- **Binaural Audio Processing**: Spatial audio using HRTFs (Head-Related Transfer Functions)
- **Speaker Embeddings**: D-vector embeddings for speaker identification
- **Real-time Processing**: Causal TF-GridNet architecture for low-latency inference

---

## Repository Structure

```
LookOnceToHear/
├── configs/                     # Training configurations
│   ├── tsh.json                # Target Speech Hearing model config
│   └── embed.json              # Speaker embedding model config
├── data/                       # Data directory (symbolic link to your data)
│   └── MixLibriSpeech/         # Main dataset
│       ├── librispeech_scaper_fmt/     # Clean speech data
│       ├── wham_noise/                 # Background noise
│       ├── jams/                       # Audio mixture specifications
│       ├── librispeech_dvector_embeddings/  # Speaker embeddings
│       └── CIPIC/                      # HRTF data
├── runs/                       # Model checkpoints and training outputs
│   ├── tsh/                    # Target Speech Hearing model
│   │   ├── best.ckpt          # Best model checkpoint
│   │   └── config.json        # Model configuration
│   └── embed/                  # Speaker embedding model
│       ├── best.ckpt          # Best embedding model
│       └── config.json        # Embedding model config
├── src/                        # Source code
│   ├── models/                 # Neural network architectures
│   │   ├── tfgridnet_realtime/ # Real-time TF-GridNet (main model)
│   │   └── tfgridnet_orig/     # Original TF-GridNet (embedding model)
│   ├── datasets/               # Data loading and processing
│   │   ├── MixLibriSpeech*.py  # Dataset classes
│   │   ├── dvector_embeddings.py      # Speaker embedding generation
│   │   ├── generate_jams.py           # Audio mixture generation
│   │   ├── multi_ch_simulator.py      # Binaural audio simulation
│   │   └── augmentations.py           # Data augmentation
│   ├── losses/                 # Loss functions
│   │   ├── sisdr_with_pit.py   # SI-SDR with Permutation Invariant Training
│   │   └── perceptual_losses.py       # Perceptual loss functions
│   ├── eval/                   # Evaluation utilities
│   ├── trainer.py              # Main training script
│   ├── ts_hear_test.py         # Evaluation script
│   └── utils.py                # Utility functions
├── requirements.txt            # Python dependencies
├── slurm.py                   # SLURM cluster job submission
└── README.md                  # Basic project information
```

### Key Components Explained

#### 1. **Models Directory**
- **`tfgridnet_realtime/`**: Main target speech hearing model
  - Causal architecture for real-time processing
  - Chunked STFT processing for low latency
  - Speaker embedding conditioning
- **`tfgridnet_orig/`**: Speaker embedding extraction model
  - Non-causal architecture for better embedding quality
  - Used during enrollment phase

#### 2. **Datasets Directory**
- **`MixLibriSpeechNoisyEnrollNorm.py`**: Main dataset class
  - Loads clean speech, noise, and HRTF data
  - Generates binaural mixtures on-the-fly
  - Handles speaker embeddings and enrollment samples
- **`multi_ch_simulator.py`**: Binaural audio simulation
  - CIPIC, RRBRIR, ASH, CATT HRTF databases
  - Motion simulation for moving speakers
- **`dvector_embeddings.py`**: Speaker embedding generation
  - Uses pre-trained d-vector model
  - Extracts 256-dimensional speaker representations

#### 3. **Losses Directory**
- **`sisdr_with_pit.py`**: Scale-Invariant Signal-to-Distortion Ratio with Permutation Invariant Training
- **`perceptual_losses.py`**: Perceptual loss functions for better audio quality

---

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB system RAM
- **Storage**: 100GB+ free space for data
- **CPU**: Multi-core processor (8+ cores recommended)

### Recommended Requirements
- **GPU**: NVIDIA RTX 4080/4090 or A100 (16GB+ VRAM)
- **RAM**: 32GB+ system RAM
- **Storage**: 500GB+ SSD storage
- **CPU**: High-end multi-core processor (16+ cores)

### Software Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.9
- **CUDA**: Compatible with PyTorch (11.8+ recommended)
- **Conda**: For environment management

---

## Installation & Setup

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n ts-hear python=3.9
conda activate ts-hear

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Setup

You need to obtain the MixLibriSpeech dataset and set up the directory structure:

#### Option A: Symbolic Link (Recommended)
If your data is at `D:\capstone\MixLibriSpeech\MixLibriSpeech`:

```bash
# Windows (run as Administrator)
mklink /D data\MixLibriSpeech D:\capstone\MixLibriSpeech\MixLibriSpeech

# Linux/macOS
ln -s /path/to/your/MixLibriSpeech data/MixLibriSpeech
```

#### Option B: Update Configuration Files
Modify paths in `configs/tsh.json` and `configs/embed.json` to point to your data location.

### Step 3: Download Pre-trained Models

Download the pre-trained model checkpoints and place them in:
- `runs/tsh/best.ckpt` - Target Speech Hearing model
- `runs/embed/best.ckpt` - Speaker embedding model

### Step 4: Generate Speaker Embeddings

**Critical Step**: Generate speaker embeddings for all audio files:

```bash
python -m src.datasets.dvector_embeddings \
    --root_dir data/MixLibriSpeech/librispeech_scaper_fmt \
    --output_dir data/MixLibriSpeech/librispeech_dvector_embeddings
```

This process:
- Extracts 256-dimensional d-vector embeddings for each speaker
- Creates embedding files organized by speaker ID
- Takes several hours to complete (depending on dataset size)

---

## Data Requirements

### Core Dataset Structure

```
data/MixLibriSpeech/
├── librispeech_scaper_fmt/              # Clean speech data (Scaper format)
│   ├── train-clean-360/                 # Training set (360 hours)
│   │   ├── [speaker_id]/
│   │   │   ├── [chapter_id]/
│   │   │   │   └── *.wav               # Audio files
│   ├── dev-clean/                       # Validation set
│   └── test-clean/                      # Test set
├── wham_noise/                          # Background noise from WHAM dataset
│   ├── tr/                              # Training noise
│   ├── cv/                              # Validation noise
│   └── tt/                              # Test noise
├── jams/                                # Audio mixture specifications
│   ├── train-clean-360/                 # Training mixtures
│   │   ├── [mixture_id]/
│   │   │   ├── mixture.jams            # Scaper specification
│   │   │   └── mixture.txt             # Mixture metadata
│   ├── dev-clean/                       # Validation mixtures
│   └── test-clean/                      # Test mixtures
├── librispeech_dvector_embeddings/      # Speaker embeddings (generated)
│   ├── train-clean-360/
│   │   └── [speaker_id].pt             # PyTorch embedding files
│   ├── dev-clean/
│   └── test-clean/
└── CIPIC/                               # HRTF data
    ├── train_hrtf.txt                  # Training HRTF file list
    ├── val_hrtf.txt                    # Validation HRTF file list
    └── test_hrtf.txt                   # Test HRTF file list
```

### Additional HRTF Datasets (Optional)

For enhanced spatial audio simulation:

```
data/
├── RRBRIR/                             # Room impulse responses
├── ASH-Listening-Set-8.0/BRIRs/        # ASH binaural room impulse responses
└── CATT_RIRs/Binaural/16k/            # CATT acoustic simulation RIRs
```

### Data Generation Pipeline

The dataset uses **Scaper** toolkit to generate synthetic mixtures:

1. **Clean Speech**: LibriSpeech corpus (16kHz, mono)
2. **Background Noise**: WHAM noise dataset
3. **Spatial Simulation**: HRTF convolution for binaural audio
4. **Mixture Generation**: On-the-fly mixing during training

---

## Training

### Training Configuration

Two models need to be trained:

1. **Speaker Embedding Model** (`configs/embed.json`)
2. **Target Speech Hearing Model** (`configs/tsh.json`)

### Training Commands

```bash
# Train speaker embedding model first
python -m src.trainer \
    --config configs/embed.json \
    --run_dir runs/embed

# Train target speech hearing model
python -m src.trainer \
    --config configs/tsh.json \
    --run_dir runs/tsh
```

### Training Parameters

#### Embedding Model (`configs/embed.json`)
- **Architecture**: TF-GridNet with embedding head
- **Loss**: Cosine similarity with margin loss
- **Batch Size**: 8
- **Learning Rate**: 5e-4
- **Epochs**: 100

#### TSH Model (`configs/tsh.json`)
- **Architecture**: Causal TF-GridNet with speaker conditioning
- **Loss**: SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- **Batch Size**: 8
- **Learning Rate**: 5e-4
- **Epochs**: 100

### Monitoring Training

Training uses **Weights & Biases (wandb)** for experiment tracking:
- Loss curves and metrics
- Audio samples during training
- Model checkpoints

### Resuming Training

```bash
# Resume from last checkpoint
python -m src.trainer \
    --config configs/tsh.json \
    --run_dir runs/tsh \
    --resume

# Resume from specific checkpoint
python -m src.trainer \
    --config configs/tsh.json \
    --run_dir runs/tsh \
    --ckpt runs/tsh/epoch-50.ckpt
```

---

## Evaluation

### Quick Test

```bash
# Run evaluation on test set with sample output
python -m src.ts_hear_test --sample
```

### Full Evaluation

```bash
# Complete evaluation on test set
python -m src.ts_hear_test --dset test
```

### Evaluation Metrics

1. **SI-SNRi (Scale-Invariant Signal-to-Noise Ratio Improvement)**
   - Measures speech enhancement quality
   - Higher values indicate better separation

2. **Cosine Similarity**
   - Measures speaker embedding accuracy
   - Higher values indicate better speaker identification

3. **Perceptual Quality Metrics**
   - PESQ (Perceptual Evaluation of Speech Quality)
   - STOI (Short-Time Objective Intelligibility)

### Output Files

Evaluation generates:
- `results_test_clean.csv`: Detailed per-sample results
- Audio samples: Enhanced target speech
- Embedding similarities: Speaker identification accuracy

---

## Model Architecture

### Target Speech Hearing Model (TF-GridNet)

```
Input: Binaural Mixture [B, 2, T] + Speaker Embedding [B, 256]
├── STFT Transform
├── TF-GridNet Blocks
│   ├── Time-Frequency Attention
│   ├── Speaker Conditioning
│   └── Temporal Modeling
├── Output Projection
└── ISTFT Transform
Output: Enhanced Binaural Speech [B, 2, T]
```

#### Key Features:
- **Causal Processing**: Real-time compatible
- **Chunked STFT**: Low-latency processing (128 samples/chunk)
- **Speaker Conditioning**: Embedding-guided separation
- **Binaural Output**: Preserves spatial information

### Speaker Embedding Model

```
Input: Binaural Enrollment [B, 2, T]
├── STFT Transform
├── TF-GridNet Encoder
├── Temporal Pooling
├── Embedding Projection
Output: Speaker Embedding [B, 256]
```

#### Training Strategy:
- **Contrastive Learning**: Positive/negative speaker pairs
- **Margin Loss**: Encourages speaker discrimination
- **Data Augmentation**: Noise and spatial variations

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in configs
"batch_size": 4,  # Instead of 8
"eval_batch_size": 6,  # Instead of 10
```

#### 2. Missing Speaker Embeddings
```bash
# Generate embeddings if missing
python -m src.datasets.dvector_embeddings \
    --root_dir data/MixLibriSpeech/librispeech_scaper_fmt \
    --output_dir data/MixLibriSpeech/librispeech_dvector_embeddings
```

#### 3. Data Path Issues
- Verify symbolic link: `ls -la data/`
- Check config paths in `configs/*.json`
- Ensure data structure matches expected format

#### 4. HRTF File Errors
```bash
# Check HRTF file lists exist
ls data/MixLibriSpeech/CIPIC/*.txt
```

#### 5. Dependency Issues
```bash
# Reinstall problematic packages
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Optimization

#### 1. Data Loading
- Increase `num_workers` in configs (up to CPU cores)
- Use SSD storage for faster I/O
- Precompute embeddings before training

#### 2. Memory Usage
- Use gradient checkpointing for large models
- Enable mixed precision training
- Reduce sequence length if needed

#### 3. Training Speed
- Use multiple GPUs with DDP
- Optimize data augmentation pipeline
- Use compiled CUDA kernels where available

### Debugging

#### 1. Audio Quality Issues
- Check input/output sample rates match
- Verify HRTF simulation parameters
- Inspect generated mixtures visually

#### 2. Training Convergence
- Monitor loss curves in wandb
- Check gradient norms
- Validate data pipeline with small subset

#### 3. Evaluation Problems
- Ensure model checkpoints exist
- Verify evaluation dataset paths
- Check embedding model compatibility

---

## Advanced Usage

### Custom Datasets

To use your own dataset:

1. Convert audio to Scaper format
2. Generate JAMS specification files
3. Extract speaker embeddings
4. Update config file paths

### Real-time Inference

For real-time applications:

```python
import torch
from src.models.tfgridnet_realtime.net import Net

# Load model
model = Net(...)
model.load_state_dict(torch.load('runs/tsh/best.ckpt'))
model.eval()

# Process audio chunks
chunk_size = 128  # samples
for audio_chunk in audio_stream:
    enhanced = model(audio_chunk, speaker_embedding)
```

### Distributed Training

For multi-GPU training:

```bash
# Single node, multiple GPUs
python -m src.trainer \
    --config configs/tsh.json \
    --run_dir runs/tsh

# Multi-node training (adjust config)
# Set appropriate world_size and rank parameters
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lookoncetohear2024,
  title={Look Once to Hear: Target Speech Hearing with Noisy Examples},
  author={[Authors]},
  booktitle={CHI 2024},
  year={2024}
}
```

---

## Support

For issues and questions:
- Check this guide first
- Search existing GitHub issues
- Create new issue with detailed description
- Include error logs and system information

**Dataset Download**: Contact authors for access to pre-processed MixLibriSpeech dataset and pre-trained models.