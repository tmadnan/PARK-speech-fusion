# A Novel Fusion Architecture for PD Detection Using Semi-Supervised Speech Embeddings

This repository contains the implementation of our projection-based fusion architecture for Parkinson’s disease (PD) detection from speech, using semi-supervised embeddings from WavLM, Wav2Vec 2.0, and ImageBind.

The method corresponds to the paper:

> **“A Novel Fusion Architecture for PD Detection Using Semi-Supervised Speech Embeddings”**  
> Tariq Adnan, Abdelrahman Abdelkader, Zipei Liu, Ekram Hossain, Sooyong Park, Md Saiful Islam, and Ehsan Hoque.

---

## 1. Overview

We build a PD screening model from a short English pangram utterance (“the quick brown fox…”). Speech is recorded via webcam in home and clinical settings, then processed as follows:

1. **Audio extraction & pangram segmentation** from video recordings (via Whisper).
2. **Feature extraction**:
   - Classical acoustic features (e.g., MFCCs, jitter, shimmer).
   - Deep SSL embeddings from:
     - **Wav2Vec 2.0**
     - **WavLM**
     - **ImageBind** (audio branch).
3. **Projection-based fusion**:
   - WavLM embeddings are projected into the ImageBind feature space.
   - Projected WavLM + ImageBind embeddings are fused and fed to a shallow ANN classifier.
4. **Model evaluation**:
   - Standard train/validation/test split.
   - Cross-environment generalization (home vs. clinic vs. PD care facility).
   - Demographic error / bias analysis.

The repository is organized to support:
- Reproducible **training & evaluation** of the fusion model from pre-extracted features.
- Future release of **feature extraction scripts** and **de-identified feature datasets**.

---

## 2. Repository Structure

At a high level:

- `code/`  
  Core Python code for:
  - Data loading and preprocessing (feature scaling, correlation filtering, optional resampling).
  - Baseline models (SVM, ANN, CNN on raw audio).
  - Projection-based fusion model (WavLM → ImageBind).
  - Training, validation, and test loops.
  - Metric computation and logging (e.g., AUROC, accuracy, sensitivity, specificity, PPV/NPV).

- `environment_full.yml`  
  Conda environment specification capturing all dependencies (Python, PyTorch with CUDA 11.8, scikit-learn, pandas, matplotlib/seaborn, Hugging Face `datasets` & `transformers` stack, W&B, etc.).

- `data/` *(planned)*  
  - De-identified, pre-extracted feature files used in the paper (classical acoustic features, Wav2Vec 2.0, WavLM, ImageBind).
  - Metadata files (e.g., participant-level labels, cohort IDs, demographic attributes) with all PHI removed.

- `notebooks/` *(optional / planned)*  
  - Example analysis notebooks (e.g., exploratory statistics, ablation summaries, error analysis visualizations).

The exact file names inside `code/` and `data/` may evolve; please refer to in-file docstrings and comments for details on usage once the public release is finalized.

---

## 3. Installation

We recommend using Conda.

```bash
# Clone the repository
git clone https://github.com/<YOUR_ORG_OR_USER>/PARK-speech-fusion.git
cd PARK-speech-fusion

# Create the environment from the exported spec
conda env create -f environment_full.yml

# Activate the environment
conda activate audio_experiment
