# Robust Audio Deepfake Detection Under Real-World Compression

A research-grade system for detecting audio deepfakes that have been compressed through real-world channels (WhatsApp, Instagram, TikTok, phone calls). This project evaluates detector robustness under realistic codec pipelines and implements techniques to improve it.

## 🎯 Project Highlights

- **Multi-Dataset Evaluation**: Trained and tested on ASVspoof 2019/2021, WaveFake, and FakeAVCeleb
- **Real-World Robustness**: Simulates 10+ compression scenarios (AAC, Opus, MP3 at various bitrates)
- **3 Detection Approaches**: 
  - Baseline Spectrogram CNN
  - Data-Augmented CNN (compression-robust)
  - Fine-tuned wav2vec2 (transfer learning)
- **Public Demo**: Interactive Gradio app deployed on HuggingFace Spaces

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the repository
cd audio-deepfake-detector

# Run automated setup (creates venv and installs dependencies)
./setup.sh

# Activate virtual environment (required for all commands)
source venv/bin/activate

# OR use the run.sh helper (auto-activates venv)
./run.sh python scripts/quickstart.py
```

**Note**: macOS Python 3.13+ requires a virtual environment. The `setup.sh` script handles this automatically.

### Download Datasets

```bash
# Make sure venv is activated: source venv/bin/activate

# Get download instructions
python scripts/download_datasets.py

# OR use helper script
./run.sh python scripts/download_datasets.py
```

### Train Models

```bash
# 1. Train baseline model (clean audio only)
python src/training/train.py --config experiments/configs/baseline.yaml

# 2. Train data-augmented model
python src/training/train.py --config experiments/configs/augmented.yaml

# 3. Fine-tune wav2vec2
python src/training/train.py --config experiments/configs/wav2vec2.yaml
```

### Run Robustness Experiments

```bash
# Evaluate all models on compressed audio variants
python src/training/evaluate.py --experiment robustness_analysis
```

### Launch Demo Locally

```bash
cd demo
python app.py
```

Visit `http://localhost:7860` to use the interface.

## 📁 Project Structure

```
audio-deepfake-detector/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed features
│   └── simulated/              # Codec-transformed variants
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Audio normalization, spectrograms
│   │   ├── codec_simulator.py  # ffmpeg pipeline simulator
│   │   └── dataset.py          # PyTorch dataset loaders
│   ├── models/
│   │   ├── baseline_cnn.py     # Spectrogram CNN
│   │   ├── augmented_cnn.py    # Data-augmented variant
│   │   └── wav2vec2_model.py   # Pretrained embedding model
│   ├── training/
│   │   ├── train.py            # Training script
│   │   └── evaluate.py         # Evaluation and visualization
│   └── utils/
│       ├── audio_utils.py      # Audio processing helpers
│       └── metrics.py          # ROC AUC, confusion matrix, etc.
├── experiments/
│   ├── configs/                # Training configurations
│   ├── notebooks/              # Analysis notebooks
│   └── results/                # Metrics, plots, checkpoints
├── demo/
│   ├── app.py                  # Gradio interface
│   └── README.md               # HuggingFace Space docs
└── tests/                      # Unit tests
```

## 🔬 Research Questions Addressed

1. **How fragile are audio deepfake detectors to realistic re-encoding?**
2. **Which codecs and bitrates are most harmful (or helpful) for detection?**
3. **Does training with realistic channel simulations improve robustness?**
4. **Are pretrained audio representations more resilient to compression?**

## 📊 Key Findings

*(To be populated after experiments)*

## 🎨 Demo Features

- Upload audio files for deepfake detection
- Simulate real-world compression (WhatsApp, Instagram, Phone, Custom)
- Compare 3 model architectures
- Visualize spectrogram and prediction confidence
- Educational explanations of results

## 🔧 Technical Details

### Codec Simulation Pipeline

- **AAC**: 128k, 64k, 32k bitrates
- **Opus**: 64k, 24k bitrates
- **MP3**: 128k, 64k, 32k bitrates
- **Sample rates**: 48kHz → 16kHz → 8kHz
- **Social media chains**: WhatsApp (16kHz mono Opus 24k), Instagram (AAC re-encode), Phone (8kHz narrowband)

### Model Architectures

**Baseline CNN**: ResNet-18 style architecture on log-mel spectrograms (128 mel bins, 1-3s windows)

**Augmented CNN**: Same architecture, trained with on-the-fly codec augmentation

**wav2vec2**: HuggingFace wav2vec2-base with fine-tuned classification head

## 📈 Results Summary

*(Results table to be added after experiments)*

## 🚢 Deployment

The demo is deployed on HuggingFace Spaces: [Link TBD]

## 📝 Citation

If you use this work, please cite:

```
[Your citation format]
```

## 📄 License

MIT License (or your preferred license)

## 🙏 Acknowledgments

- ASVspoof Challenge organizers
- WaveFake dataset creators
- FakeAVCeleb dataset creators
- HuggingFace for wav2vec2 and Spaces hosting
