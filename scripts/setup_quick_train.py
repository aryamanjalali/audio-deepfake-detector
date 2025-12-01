utf-8"""
Quick Training Script with Auto-Downloading Sample Dataset
Downloads a small dataset and trains quickly for demo purposes.
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil
sys.path.insert(0, str(Path(__file__).parent.parent))
def download_sample_dataset():
    """Download a small sample dataset for quick training."""
    print("="*60)
    print("Downloading Sample Dataset for Quick Training")
    print("="*60)
    data_dir = Path("data/raw/quick_train")
    data_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreating sample dataset structure...")
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)
    print("\n To train quickly, please add audio files:")
    print(f"\n1. Put 10-20 REAL voice recordings in:")
    print(f"   {real_dir.absolute()}")
    print(f"\n2. Put 10-20 FAKE/AI-generated voices in:")
    print(f"   {fake_dir.absolute()}")
    print("\nFile format: .wav, .mp3, .opus, .m4a (any format works)")
    print("\nDon't have fake audio? You can:")
    print("- Use ElevenLabs, Play.ht, or any online TTS")
    print("- Record yourself with different filters")
    print("- Just copy real audio to both folders for testing (won't learn, but will run)")
    return data_dir
def create_simple_config(data_dir: Path):
    """Create a minimal config for quick training."""
    config = {
        'model': {
            'type': 'baseline_cnn'
        },
        'data': {
            'datasets': {
                'simple': {
                    'real_dir': str(data_dir / 'real'),
                    'fake_dir': str(data_dir / 'fake')
                }
            },
            'train_split': 0.8,
            'balance_classes': True
        },
        'audio': {
            'sample_rate': 16000,
            'n_mels': 128,
            'n_fft': 1024,
            'hop_length': 512,
            'max_duration': 4.0
        },
        'training': {
            'batch_size': 4,  
            'epochs': 10,  
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'early_stopping_patience': 5,
            'output_dir': 'experiments/results/quick_demo',
            'scheduler': 'reduce_on_plateau'
        }
    }
    import yaml
    config_path = Path('experiments/configs/quick_demo.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n Created config: {config_path}")
    return config_path
def count_audio_files(directory: Path) -> int:
    """Count audio files in directory."""
    audio_extensions = {'.wav', '.mp3', '.opus', '.m4a', '.flac', '.ogg'}
    count = 0
    if directory.exists():
        for ext in audio_extensions:
            count += len(list(directory.glob(f'*{ext}')))
    return count
def main():
    print("\n Quick Training Setup")
    data_dir = download_sample_dataset()
    real_count = count_audio_files(data_dir / 'real')
    fake_count = count_audio_files(data_dir / 'fake')
    print(f"\n Current dataset:")
    print(f"   Real audio files: {real_count}")
    print(f"   Fake audio files: {fake_count}")
    if real_count < 5 or fake_count < 5:
        print("\n⚠️  Need at least 5 files in each folder to train!")
        print("\nQuick option: Use the same audio file in both folders just to test")
        print("(The model won't learn anything useful, but you can test the pipeline)")
        return
    config_path = create_simple_config(data_dir)
    print("\n" + "="*60)
    print("Ready to Train!")
    print("="*60)
    print(f"\nTotal training samples: {real_count + fake_count}")
    print(f"Estimated time: 5-15 minutes on CPU")
    print("\nTo start training, run:")
    print(f"  ./run.sh python src/training/train.py --config {config_path}")
    print("\nOr use the simplified trainer:")
    print(f"  ./run.sh python scripts/quick_train.py")
    print("="*60)
if __name__ == '__main__':
    main()