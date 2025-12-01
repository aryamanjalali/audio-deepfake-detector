utf-8"""
Download scripts for datasets.
Run this file to download all three datasets:
- ASVspoof 2019 LA
- WaveFake  
- FakeAVCeleb
"""
from pathlib import Path
import sys
def download_asvspoof():
    """
    Download ASVspoof 2019 LA dataset.
    Note: Requires registration at https://datashare.ed.ac.uk/handle/10283/3336
    This function provides instructions for manual download.
    """
    print("\n" + "="*60)
    print("ASVspoof 2019 LA Dataset")
    print("="*60)
    print("\nASVspoof requires manual registration and download.")
    print("\nSteps:")
    print("1. Go to: https://datashare.ed.ac.uk/handle/10283/3336")
    print("2. Register and download the following files:")
    print("   - LA.zip (Training and development)")
    print("   - LA.eval.zip (Evaluation set)")
    print("3. Extract to: data/raw/ASVspoof2019/LA/")
    print("\nExpected structure:")
    print("data/raw/ASVspoof2019/LA/")
    print("  ├── ASVspoof2019_LA_train/")
    print("  ├── ASVspoof2019_LA_dev/")
    print("  ├── ASVspoof2019_LA_eval/")
    print("  └── ASVspoof2019_LA_cm_protocols/")
    print("="*60)
def download_wavefake():
    """
    Download WaveFake dataset.
    Note: WaveFake is hosted on Zenodo
    """
    print("\n" + "="*60)
    print("WaveFake Dataset")
    print("="*60)
    print("\nWaveFake requires manual download from Zenodo.")
    print("\nSteps:")
    print("1. Go to: https://zenodo.org/record/5642694")
    print("2. Download wavefake_v1.zip")
    print("3. Extract to: data/raw/WaveFake/")
    print("\nExpected structure:")
    print("data/raw/WaveFake/")
    print("  ├── generated_audio/")
    print("  │   ├── <vocoder1>/")
    print("  │   ├── <vocoder2>/")
    print("  │   └── ...")
    print("  └── real_audio/")
    print("      └── ljspeech/")
    print("="*60)
def download_fakeavceleb():
    """
    Download FakeAVCeleb dataset.
    Note: FakeAVCeleb requires registration
    """
    print("\n" + "="*60)
    print("FakeAVCeleb Dataset")
    print("="*60)
    print("\nFakeAVCeleb requires registration.")
    print("\nSteps:")
    print("1. Go to: https://sites.google.com/view/fakeavcelebdash-lab/")
    print("2. Fill out the registration form")
    print("3. Download the dataset after approval")
    print("4. Extract to: data/raw/FakeAVCeleb/")
    print("\nExpected structure:")
    print("data/raw/FakeAVCeleb/")
    print("  ├── RealVideo-RealAudio/")
    print("  ├── FakeVideo-FakeAudio/")
    print("  └── RealVideo-FakeAudio/")
    print("="*60)
def main():
    """Main download function."""
    print("="*60)
    print("Audio Deepfake Dataset Downloader")
    print("="*60)
    print("\nThis script will guide you through downloading the datasets.")
    print("Note: All datasets require manual download due to licensing.")
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    download_asvspoof()
    download_wavefake()
    download_fakeavceleb()
    print("\n" + "="*60)
    print("Download Instructions Complete")
    print("="*60)
    print("\nAfter downloading all datasets, verify the structure with:")
    print("  python scripts/verify_datasets.py")
    print("\nThen proceed to training:")
    print("  python src/training/train.py --config experiments/configs/baseline.yaml")
    print("="*60)
if __name__ == "__main__":
    main()