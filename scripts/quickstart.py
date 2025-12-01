utf-8"""
Quick Start Guide Script
This script helps you get started with the project.
"""
import subprocess
import sys
from pathlib import Path
def print_banner(text):
    """Print a banner."""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")
def check_dependencies():
    """Check if required dependencies are installed."""
    print_banner("Checking Dependencies")
    try:
        import torch
        print(f" PyTorch {torch.__version__}")
    except ImportError:
        print(" PyTorch not installed")
        return False
    try:
        import torchaudio
        print(f" Torchaudio {torchaudio.__version__}")
    except ImportError:
        print(" Torchaudio not installed")
        return False
    try:
        import gradio
        print(f" Gradio {gradio.__version__}")
    except ImportError:
        print(" Gradio not installed")
        return False
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f" {version}")
        else:
            print(" ffmpeg not working properly")
            return False
    except FileNotFoundError:
        print(" ffmpeg not installed")
        print("  Install with: brew install ffmpeg (macOS)")
        return False
    return True
def test_codec_simulator():
    """Test the codec simulator."""
    print_banner("Testing Codec Simulator")
    try:
        from src.data.codec_simulator import CodecSimulator
        simulator = CodecSimulator()
        print(" Codec simulator initialized")
        print(f"  Available codecs: {list(CodecSimulator.CODECS.keys())}")
        print(f"  Social media chains: {list(CodecSimulator.SOCIAL_MEDIA_CHAINS.keys())}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False
def test_models():
    """Test model architectures."""
    print_banner("Testing Model Architectures")
    try:
        import torch
        from src.models.baseline_cnn import BaselineCNN
        from src.models.augmented_cnn import AugmentedCNN
        model = BaselineCNN(n_mels=128, num_classes=2)
        dummy_input = torch.randn(2, 1, 128, 256)
        output = model(dummy_input)
        params = sum(p.numel() for p in model.parameters())
        print(f" Baseline CNN - {params:,} parameters")
        model = AugmentedCNN(n_mels=128, num_classes=2)
        output = model(dummy_input)
        print(f" Augmented CNN - Same architecture")
        try:
            from src.models.wav2vec2_model import Wav2Vec2Classifier
            print(" wav2vec2 model available")
            print("  (Note: Will download ~360MB on first use)")
        except Exception as e:
            print(f"⚠ wav2vec2 not tested: {e}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False
def run_demo():
    """Run the Gradio demo."""
    print_banner("Launching Gradio Demo")
    print("Starting demo server...")
    print("The demo will open at http://localhost:7860")
    print("\nPress Ctrl+C to stop the server")
    print("\nNote: Models will use random weights until trained.")
    try:
        subprocess.run([sys.executable, 'demo/app.py'])
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
def show_next_steps():
    """Show next steps."""
    print_banner("Next Steps")
    print("1. Download Datasets:")
    print("   python scripts/download_datasets.py")
    print()
    print("2. Train Baseline Model:")
    print("   python src/training/train.py --config experiments/configs/baseline.yaml")
    print()
    print("3. Train Augmented Model:")
    print("   python src/training/train.py --config experiments/configs/augmented.yaml")
    print()
    print("4. Train wav2vec2 Model:")
    print("   python src/training/train.py --config experiments/configs/wav2vec2.yaml")
    print()
    print("5. Run Robustness Experiments:")
    print("   python src/training/evaluate.py --experiment robustness")
    print()
    print("6. Launch Demo:")
    print("   python demo/app.py")
    print()
    print("="*60)
def main():
    """Main function."""
    print_banner("Audio Deepfake Detector - Quick Start")
    if not check_dependencies():
        print("\n Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return
    test_codec_simulator()
    test_models()
    print_banner("What would you like to do?")
    print("1. Show next steps (recommended for first time)")
    print("2. Launch demo (with untrained models)")
    print("3. Exit")
    choice = input("\nEnter choice (1-3): ").strip()
    if choice == '1':
        show_next_steps()
    elif choice == '2':
        run_demo()
    else:
        print("\nGoodbye!")
if __name__ == "__main__":
    main()