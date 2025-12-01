#!/bin/bash

echo "=============================================="
echo "Audio Deepfake Detection - Command Reference"
echo "=============================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SETUP COMMANDS (Run Once) ===${NC}"
echo ""

echo "1. Install ffmpeg (required for codec simulation):"
echo "   brew install ffmpeg"
echo ""

echo "2. Setup virtual environment (already done):"
echo "   ./setup.sh"
echo ""

echo -e "${BLUE}=== ACTIVATE VIRTUAL ENVIRONMENT (Every Session) ===${NC}"
echo ""

echo "Activate venv (required before running Python commands):"
echo "   source venv/bin/activate"
echo ""

echo -e "${BLUE}=== GETTING STARTED ===${NC}"
echo ""

echo "1. View dataset download instructions:"
echo "   ./run.sh python scripts/download_datasets.py"
echo "   OR: python scripts/download_datasets.py (after activating venv)"
echo ""

echo "2. Run quick start guide:"
echo "   ./run.sh python scripts/quickstart.py"
echo ""

echo -e "${BLUE}=== DEMO COMMANDS ===${NC}"
echo ""

echo "1. Launch Gradio demo locally (no training required):"
echo "   ./run.sh python demo/app.py"
echo "   Then open: http://localhost:7860"
echo ""

echo "2. Launch with public sharing:"
echo "   ./run.sh python demo/app.py --share"
echo ""

echo -e "${BLUE}=== TRAINING COMMANDS (Optional - Requires Datasets) ===${NC}"
echo ""

echo "1. Train baseline model (~4-6 hours on GPU):"
echo "   ./run.sh python src/training/train.py --config experiments/configs/baseline.yaml"
echo ""

echo "2. Train augmented model (~4-6 hours on GPU):"
echo "   ./run.sh python src/training/train.py --config experiments/configs/augmented.yaml"
echo ""

echo "3. Train wav2vec2 model (~8-10 hours on GPU):"
echo "   ./run.sh python src/training/train.py --config experiments/configs/wav2vec2.yaml"
echo ""

echo -e "${BLUE}=== EVALUATION COMMANDS (After Training) ===${NC}"
echo ""

echo "1. Run robustness experiments:"
echo "   ./run.sh python src/training/evaluate.py --experiment robustness"
echo ""

echo "2. Evaluate specific model on test set:"
echo "   ./run.sh python src/training/evaluate.py --model baseline --checkpoint experiments/results/baseline/best_model.pth"
echo ""

echo -e "${BLUE}=== TEST INDIVIDUAL COMPONENTS ===${NC}"
echo ""

echo "1. Test codec simulator:"
echo "   ./run.sh python -m src.data.codec_simulator"
echo ""

echo "2. Test audio preprocessing:"
echo "   ./run.sh python -m src.data.preprocessing"
echo ""

echo "3. Test baseline model:"
echo "   ./run.sh python -m src.models.baseline_cnn"
echo ""

echo "4. Test augmented model:"
echo "   ./run.sh python -m src.models.augmented_cnn"
echo ""

echo "5. Test wav2vec2 model (downloads ~360MB on first run):"
echo "   ./run.sh python -m src.models.wav2vec2_model"
echo ""

echo -e "${BLUE}=== DEPLOYMENT TO HUGGINGFACE SPACES ===${NC}"
echo ""

echo "1. Login to HuggingFace:"
echo "   huggingface-cli login"
echo ""

echo "2. Create new space:"
echo "   huggingface-cli repo create audio-deepfake-detector --type space --space_sdk gradio"
echo ""

echo "3. Deploy (from demo directory):"
echo "   cd demo"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial deployment'"
echo "   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/audio-deepfake-detector"
echo "   git push space main"
echo ""

echo -e "${BLUE}=== UTILITY COMMANDS ===${NC}"
echo ""

echo "1. Check project structure:"
echo "   tree -L 2 -I 'venv|__pycache__|*.pyc'"
echo ""

echo "2. View installed packages:"
echo "   ./run.sh pip list"
echo ""

echo "3. Deactivate virtual environment:"
echo "   deactivate"
echo ""

echo -e "${BLUE}=== QUICK EXAMPLE WORKFLOWS ===${NC}"
echo ""

echo -e "${YELLOW}Workflow 1: Test the demo immediately (no datasets needed)${NC}"
echo "   ./run.sh python demo/app.py"
echo ""

echo -e "${YELLOW}Workflow 2: Full training pipeline${NC}"
echo "   1. Install ffmpeg: brew install ffmpeg"
echo "   2. Download datasets: ./run.sh python scripts/download_datasets.py"
echo "   3. Train baseline: ./run.sh python src/training/train.py --config experiments/configs/baseline.yaml"
echo "   4. Train augmented: ./run.sh python src/training/train.py --config experiments/configs/augmented.yaml"
echo "   5. Train wav2vec2: ./run.sh python src/training/train.py --config experiments/configs/wav2vec2.yaml"
echo "   6. Run experiments: ./run.sh python src/training/evaluate.py --experiment robustness"
echo "   7. Launch demo: ./run.sh python demo/app.py"
echo ""

echo -e "${YELLOW}Workflow 3: Deploy to HuggingFace${NC}"
echo "   1. Train models (see Workflow 2)"
echo "   2. Copy trained models to demo directory"
echo "   3. huggingface-cli login"
echo "   4. Deploy (see deployment commands above)"
echo ""

echo "=============================================="
echo -e "${GREEN} All commands listed above!${NC}"
echo "=============================================="
echo ""
echo " TIP: Use ./run.sh before any Python command to auto-activate venv"
echo " For more details, see:"
echo "   - START_HERE.md"
echo "   - QUICKSTART.md"
echo "   - README.md"
echo ""
