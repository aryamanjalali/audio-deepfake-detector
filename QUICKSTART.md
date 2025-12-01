# 🎯 Quick Start Guide

Everything is now ready to use! Just follow these simple steps:

## 1. Setup (First Time Only - Already Done! ✅)

The virtual environment is set up and all dependencies are installed.

## 2. Activate Virtual Environment

Every time you use the project (in a new terminal), activate the virtual environment:

```bash
cd /Users/aryaman/Desktop/Projects/audio-deepfake-detector
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal prompt.

## 3. Available Commands

### Get Dataset Download Instructions
```bash
python scripts/download_datasets.py
```

### Run Quick Start Guide
```bash
python scripts/quickstart.py
```

### Launch Demo (Without Training)
```bash
python demo/app.py
```
Then open http://localhost:7860 in your browser

## Alternative: Use Helper Script

You can use `./run.sh` to automatically activate the venv:

```bash
./run.sh python scripts/download_datasets.py
./run.sh python scripts/quickstart.py
./run.sh python demo/app.py
```

## When You're Done

To deactivate the virtual environment:
```bash
deactivate
```

## Next Steps (After Dataset Download)

1. **Download datasets** - Follow instructions from `python scripts/download_datasets.py`
2. **Train models** - See training configs in `experiments/configs/`
3. **Run experiments** - Evaluate robustness across codec chains
4. **Deploy demo** - Push to HuggingFace Spaces

## Troubleshooting

**Q: Command not found?**
A: Make sure venv is activated: `source venv/bin/activate`

**Q: Module not found?**
A: Run `./setup.sh` again to reinstall dependencies

**Q: ffmpeg not found?**
A: Install with `brew install ffmpeg`
