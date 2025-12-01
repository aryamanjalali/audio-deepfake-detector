# Audio Deepfake Training Guide

## Quick Start (With Sample Data)

Since downloading full datasets takes time, here's how to train on a small sample to test the system:

### Option 1: Use Sample Audio Files (Quick Test)

1. **Create a small test dataset:**
```bash
mkdir -p data/raw/test_data/{real,fake}
```

2. **Add some audio files:**
   - Put 5-10 real voice recordings in `data/raw/test_data/real/`
   - Put 5-10 AI-generated voices in `data/raw/test_data/fake/`
   - (Or just use the same files in both for testing - predictions will be random but training will work)

3. **Update config to use test data:**
   Edit `experiments/configs/baseline.yaml` and change the data paths to point to your test data

4. **Train:**
```bash
./run.sh python src/training/train.py --config experiments/configs/baseline.yaml
```

### Option 2: Download Full Datasets (Best Results)

This gives you proper trained models but takes longer:

#### Step 1: Download Datasets

Run the download helper:
```bash
./run.sh python scripts/download_datasets.py
```

This will show you where to download:
- **ASVspoof 2019 LA** (10GB): https://datashare.ed.ac.uk/handle/10283/3336
- **WaveFake** (4GB): https://zenodo.org/record/5642694
- **FakeAVCeleb** (varies): https://sites.google.com/view/fakeavcelebdash-lab/

#### Step 2: Organize Data

Extract datasets to:
```
data/raw/
├── ASVspoof2019/LA/
├── WaveFake/
└── FakeAVCeleb/
```

#### Step 3: Train Models

```bash
# Baseline (fastest, ~2-3 hours on GPU)
./run.sh python src/training/train.py --config experiments/configs/baseline.yaml

# Augmented (robust to codecs, ~3-4 hours)
./run.sh python src/training/train.py --config experiments/configs/augmented.yaml

# wav2vec2 (best performance, ~6-8 hours)
./run.sh python src/training/train.py --config experiments/configs/wav2vec2.yaml
```

## Training on CPU vs GPU

- **GPU**: 2-8 hours per model (recommended)
- **CPU**: 24-48 hours per model (very slow, but works)
- **MPS (Mac M1/M2)**: 4-12 hours per model

## What Happens During Training

1. **Loads data** from configured datasets
2. **Preprocesses audio** (converts to spectrograms or embeddings)
3. **Trains model** for specified epochs
4. **Validates** on held-out data each epoch
5. **Saves checkpoints** at `experiments/results/{model_name}/`
6. **Tracks metrics**: accuracy, ROC AUC, loss

## After Training

1. **Restart the demo:**
```bash
pkill -f "demo/app.py"
./run.sh python demo/app.py
```

2. **Test predictions** - should now show real confidence scores!

3. **Check training curves:**
```bash
open experiments/results/baseline/training_curves.png
```

## Common Issues

**"Error creating dataloaders"**
→ Datasets not downloaded yet. Use Option 1 (test data) or download datasets

**"CUDA out of memory"**
→ Reduce batch_size in config file (try 16 or 8)

**Training very slow on CPU**
→ Expected! Consider using Google Colab with free GPU

## Expected Results

After training on full datasets:

| Model | Clean Accuracy | WhatsApp Accuracy |
|-------|---------------|-------------------|
| Baseline | ~92% | ~75% |
| Augmented | ~91% | ~87% |
| wav2vec2 | ~94% | ~90% |

**Ready to help you train! Which option would you like to try first?**
