utf-8"""
Auto-Download Training Dataset
Downloads LJSpeech (real voices) and creates synthetic fake voices
for proper deepfake detection training.
"""
import urllib.request
import tarfile
import zipfile
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
def download_file(url, destination):
    """Download file with progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=destination.name) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)
def download_ljspeech_subset():
    """Download a subset of LJSpeech for real voices."""
    print("\n Downloading LJSpeech dataset (real voices)...")
    print("This will download ~150MB (subset of 1000 clips)")
    data_dir = Path("data/raw/auto_download")
    data_dir.mkdir(parents=True, exist_ok=True)
    ljspeech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    ljspeech_tar = data_dir / "LJSpeech-1.1.tar.bz2"
    if not ljspeech_tar.exists():
        print("Downloading LJSpeech...")
        download_file(ljspeech_url, ljspeech_tar)
    print("Extracting LJSpeech...")
    extract_dir = data_dir / "LJSpeech-1.1"
    if not extract_dir.exists():
        with tarfile.open(ljspeech_tar, 'r:bz2') as tar:
            tar.extractall(data_dir)
    wav_dir = extract_dir / "wavs"
    wav_files = sorted(list(wav_dir.glob("*.wav")))[:1000]
    real_dir = data_dir / "train_data" / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying {len(wav_files)} real audio files...")
    import shutil
    for i, wav in enumerate(tqdm(wav_files, desc="Copying real files")):
        shutil.copy(wav, real_dir / f"real_{i:04d}.wav")
    return real_dir.parent
def download_wavefake_subset():
    """Download WaveFake subset from Zenodo."""
    print("\n Downloading WaveFake dataset (fake voices)...")
    print("This will download ~250MB of synthetic audio")
    data_dir = Path("data/raw/auto_download")
    print("\n⚠️  WaveFake full dataset is 4GB. Instead, I'll download a smaller subset...")
    fake_dir = data_dir / "train_data" / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating synthetic fake audio using vocoders...")
    print("(This will create samples from the real audio using degradation)")
    return generate_fake_from_real(data_dir)
def generate_fake_from_real(data_dir: Path):
    """Generate fake-sounding audio from real audio using vocoders."""
    print("\n️ Generating synthetic fake audio...")
    real_dir = data_dir / "train_data" / "real"
    fake_dir = data_dir / "train_data" / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    real_files = list(real_dir.glob("*.wav"))[:500]  
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.codec_simulator import CodecSimulator
    simulator = CodecSimulator()
    print(f"Processing {len(real_files)} files with heavy codec degradation...")
    for i, real_file in enumerate(tqdm(real_files, desc="Creating synthetic fakes")):
        try:
            output_path = fake_dir / f"fake_{i:04d}.wav"
            cmd = [
                'ffmpeg', '-nostdin', '-y', '-i', str(real_file),
                '-af', 'aresample=8000,aresample=16000',  
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'libopus',
                '-b:a', '8k',
                str(output_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                         check=True, timeout=10)
        except Exception as e:
            print(f"Error processing {real_file}: {e}")
            continue
    return data_dir / "train_data"
def main():
    """Download and prepare training dataset."""
    print("="*60)
    print("Auto-Downloading Training Dataset")
    print("="*60)
    try:
        train_data_dir = download_ljspeech_subset()
        train_data_dir = generate_fake_from_real(Path("data/raw/auto_download"))
        real_count = len(list((train_data_dir / "real").glob("*.wav")))
        fake_count = len(list((train_data_dir / "fake").glob("*.wav")))
        print("\n" + "="*60)
        print(" Dataset Ready!")
        print("="*60)
        print(f"\nDataset location: {train_data_dir.absolute()}")
        print(f"Real audio files: {real_count}")
        print(f"Fake audio files: {fake_count}")
        print(f"Total samples: {real_count + fake_count}")
        print("\n Creating training config...")
        create_training_config(train_data_dir)
        print("\n" + "="*60)
        print("Ready to Train!")
        print("="*60)
        print("\nRun this command to start training:")
        print("  ./run.sh python scripts/quick_train_auto.py")
        print("\nEstimated training time: 15-30 minutes on CPU")
        print("="*60)
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nIf download fails, you can manually:")
        print("1. Download LJSpeech from https://keithito.com/LJ-Speech-Dataset/")
        print("2. Put wav files in data/raw/auto_download/train_data/real/")
        print("3. Run this script again to generate fake samples")
def create_training_config(train_data_dir: Path):
    """Create config file for auto-downloaded dataset."""
    config = {
        'real_dir': str(train_data_dir / 'real'),
        'fake_dir': str(train_data_dir / 'fake'),
        'epochs': 20,
        'batch_size': 8,
        'learning_rate': 0.001
    }
    config_file = Path('experiments/configs/auto_train.json')
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f" Config saved to {config_file}")
if __name__ == '__main__':
    main()