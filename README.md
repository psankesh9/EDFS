# Emotional Vocalization Classification

[![UC Berkeley](https://img.shields.io/badge/UC%20Berkeley-DS207-003262?style=for-the-badge&labelColor=003262&color=FDB515)](https://www.berkeley.edu/)
[![MIDS](https://img.shields.io/badge/MIDS-MIDS-1E4D2B?style=for-the-badge&labelColor=002855&color=1E4D2B)](https://datascience.berkeley.edu/)
[![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-0A9396?style=for-the-badge&logo=google-scholar&logoColor=white)](#)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-FF9900?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![librosa](https://img.shields.io/badge/librosa-AF75FF?style=for-the-badge&labelColor=4C1D95&color=AF75FF)](https://librosa.org/)
[![Audio Emotion Recognition](https://img.shields.io/badge/Audio%20Emotion%20Recognition-RAVDESS-6A4C93?style=for-the-badge&labelColor=2D1B69&color=6A4C93)](https://doi.org/10.1371/journal.pone.0196391)

## Abstract

This project investigates supervised learning pipelines for recognizing affective states from vocal expressions. Using the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), we extract engineered acoustic descriptors and evaluate classical and deep models to classify emotions in acted speech and song. The work is part of the DS207 Fall 2025 course sequence at UC Berkeley and targets a deployable research baseline that will later support emotionally aware text-to-speech systems.

## Motivation

Emotion carries non-verbal cues that shape human communication, yet contemporary human-computer interaction systems still react primarily to lexical content. Accurately classifying vocal emotions can enrich online therapy, customer-support triaging, accessibility tools, and human-robot interfaces. Our team views this milestone as foundational preparation for future text-to-speech research that requires reliable emotion conditioning.

## Dataset

- **Source:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- **Scope:** 1,440 speech recordings from 24 actors across eight emotions (neutral, calm, happy, sad, angry, fearful, surprise, disgust) with labeled intensity, statement, repetition, modality, and actor identifiers.
- **Balance:** Speech recordings are balanced across speakers and emotions, with the exception of the neutral class (96 samples versus 192 for other emotions).
- **Access:** Follow the dataset license requirements at <https://zenodo.org/record/1188976>. Raw audio files are not redistributed in this repository; store them under a local `data/` directory.
- **License:** CC BY-NC-SA 4.0 (non-commercial use only).

### Dataset Download Script

This repository includes an automated download script supporting three sources:

| Source | Command | Files | Notes |
|--------|---------|-------|-------|
| **Zenodo** (default) | `python scripts/download_ravdess.py` | 1,440 (speech) or 2,452 (both) | Canonical source with integrity verification |
| **Hugging Face** | `python scripts/download_ravdess.py --source huggingface` | 1,440 | Faster, cached downloads (speech only) |
| **Deep Lake** | `python scripts/download_ravdess.py --source deeplake` | 1,440 | Streaming support (speech only) |

**Quick Start:**

```bash
# Install dependencies and download dataset
uv sync
python scripts/download_ravdess.py
```

**All Options:**

```bash
python scripts/download_ravdess.py --help

Options:
  --source {zenodo,huggingface,deeplake}  Download source (default: zenodo)
  --subset {speech,song,both}             Subset to download (default: speech)
  --output PATH                           Custom output directory
  --force                                 Overwrite existing files
  --license                               Display CC BY-NC-SA 4.0 license notice
  --quiet                                 Suppress progress bars
```

**Alternative Sources (require optional dependencies):**

```bash
# Install optional dependencies for Hugging Face or Deep Lake
uv pip install -e '.[download]'

# Then use alternative source
python scripts/download_ravdess.py --source huggingface
```

**Output Structure:**

```text
data/raw/ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_24/
│   └── ...
└── manifest.json          # Download metadata
```

## Data Preparation Pipeline

1. **File Parsing:** Programmatically decode the seven-part filename schema (e.g., `03-01-08-02-02-01-12.wav`) to recover emotion, intensity, statement, repetition, modality, and actor metadata.
2. **Standardization:** Resample each waveform from 48 kHz to 22 kHz, trim leading/trailing silence, and peak-normalize amplitudes to 1.0.
3. **Feature Extraction:**
   - 40 Mel-Frequency Cepstral Coefficients (MFCCs)
   - Log-scaled mel spectrograms
   - 12-dimensional chroma features
4. **Artifacts:** Intermediate features are currently materialized within notebooks; scripted extraction is planned for subsequent milestones.

## Exploratory Analysis Highlights

- Gender parity (12 male, 12 female actors) and controlled studio conditions support unbiased baseline training yet may hinder generalization to in-the-wild speech.
- Acoustic descriptors show distinct emotion-specific patterns; intensity labels modulate feature amplitudes predictably.
- Visualizations (see `notebooks/`) validate that MFCC grids and spectrogram textures contain separable structure for the targeted emotions.

## Anticipated Challenges and Mitigations

- **Limited speaker diversity:** Risk of overfitting to actor identities. Mitigation: enforce speaker-independent splits and consider domain adaptation data augmentation.
- **Acted emotion bias:** Exaggerated expressions may not transfer to spontaneous speech. Mitigation: evaluate on supplementary corpora and incorporate augmentation (noise injection, pitch shifts).
- **Small sample counts:** Limited data per class can destabilize deep models. Mitigation: start with classical baselines (SVM, XGBoost) and augment input representations before scaling to CNNs.

## Planned Modeling Experiments

| Model                        | Input Representation                    | Objective                                       |
| ---------------------------- | --------------------------------------- | ----------------------------------------------- |
| Support Vector Machine       | Aggregated MFCC statistics              | Establish reproducible baseline                 |
| XGBoost                      | Summarized spectral and chroma features | Assess ensemble performance on tabular features |
| Convolutional Neural Network | 2D log-mel spectrograms                 | Learn hierarchical temporal-spectral patterns   |

All models will share speaker-independent splits (80% train, 10% validation, 10% hold-out test). Evaluation will report overall accuracy, macro-averaged precision/recall/F1, and confusion matrices per emotion. The best-performing model will be stress-tested with augmentation (noise, pitch modulation) and compared against song modality extensions in later work.

## Repository Layout

```text
.
├── notebooks/   # Jupyter notebooks covering EDA, preprocessing, and prototype modeling
├── reports/     # Milestone reports and documentation artifacts
├── scripts/     # Utility scripts (dataset download, preprocessing)
└── tests/       # Unit tests
```

Add a local `data/` directory (ignored by Git) to host licensed audio files and derived features.

## Getting Started

1. **Install uv (recommended workflow)**  
   uv offers a fast, unified interface for Python versions, virtual environments, and dependency management. Refer to the official installation guide for platform-specific notes and verification steps ([docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)).

   - macOS / Linux / WSL:

     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - Windows (PowerShell):

     ```powershell
     powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

     Ensure the installer’s target directory (typically `~/.local/bin` on Unix-like systems) is present on your `PATH` before proceeding.

2. **Provision a project environment with uv**  
   Optionally install a specific Python interpreter if it is not already available locally:

   ```bash
   uv python install 3.11
   ```

   Create an isolated virtual environment in the repository root and activate it:

   ```bash
   uv venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install project dependencies**

   This project uses `pyproject.toml` for dependency management. Install all dependencies with:

   ```bash
   uv sync
   ```

   For development dependencies (pytest, ruff, pylint):

   ```bash
   uv sync --extra dev
   ```

   **Alternative: pip-only environment**

   If `uv` is unavailable, use Python's built-in tooling:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .                 # Install from pyproject.toml
   ```

4. **Acquire data assets**

   Download the RAVDESS dataset using the provided script:

   ```bash
   python scripts/download_ravdess.py
   ```

   This downloads ~200 MB of audio files to `data/raw/ravdess/`. See [Dataset Download Script](#dataset-download-script) for all options including alternative sources and custom output directories.

   **Manual Alternative:** Download directly from [Zenodo](https://zenodo.org/record/1188976) and extract to `data/raw/ravdess/`.
5. **Run the notebooks**  
   Execute the preprocessing and EDA notebooks in the following order:
   1. `Final_Project_Mileston_Draft_v1d.ipynb` – feature engineering pipeline
   2. `eda_combined_final.ipynb` – summary visualizations
   3. Specialized notebooks (`eda_pratheek.ipynb`, `Final_Project_Mileston_Draft_NithyaEDA.ipynb`) – targeted analyses

## Milestone Roadmap

- **Milestone 1 (current):** Dataset ingestion, preprocessing pipeline, exploratory analysis, model planning.
- **Milestone 2:** Implement baselines (SVM, XGBoost), begin CNN prototyping, introduce augmentation.
- **Milestone 3:** Integrate additional datasets, refine evaluation suite, prepare final report and demo.

## Team

- Nithya Srinivasan — overall approach, data preparation
- Kris Mehra — exploratory data analysis, reporting
- Bjorn Melin — model selection strategy, planning next steps
- Pratheek Sankeshi — risk assessment, documentation review

## Citation

Please cite RAVDESS when publishing results:

```text
Livingstone, S.R., & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE, 13*(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
```

---

For project updates and collaboration, refer to this repository: <https://github.com/psankesh9/EDFS>.
