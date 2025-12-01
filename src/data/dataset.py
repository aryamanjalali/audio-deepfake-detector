"""
PyTorch Dataset Classes for Audio Deepfake Detection

Supports multiple datasets:
- ASVspoof 2019/2021
- WaveFake
- FakeAVCeleb
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Union, List, Dict
import json
from .preprocessing import AudioPreprocessor


class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA (Logical Access) dataset.
    
    Dataset structure:
        ASVspoof2019/LA/
            ASVspoof2019_LA_train/
            ASVspoof2019_LA_dev/
            ASVspoof2019_LA_eval/
            ASVspoof2019_LA_cm_protocols/
                ASVspoof2019.LA.cm.train.trn.txt
                ASVspoof2019.LA.cm.dev.trl.txt
                ASVspoof2019.LA.cm.eval.trl.txt
    """
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 subset: str = 'train',
                 preprocessor: Optional[AudioPreprocessor] = None,
                 max_duration: Optional[float] = None,
                 transform: Optional[Callable] = None):
        """
        Initialize ASVspoof dataset.
        
        Args:
            root_dir: Path to ASVspoof2019/LA directory
            subset: 'train', 'dev', or 'eval'
            preprocessor: AudioPreprocessor instance
            max_duration: Maximum audio duration in seconds
            transform: Additional transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.max_duration = max_duration
        self.transform = transform
        
        # Load protocol file
        protocol_file = self.root_dir / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{subset}.trn.txt'
        if not protocol_file.exists():
            # Try .trl extension for dev/eval
            protocol_file = self.root_dir / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{subset}.trl.txt'
        
        # Parse protocol file
        self.samples = self._load_protocol(protocol_file)
        
        # Audio directory
        self.audio_dir = self.root_dir / f'ASVspoof2019_LA_{subset}' / 'flac'
        
    def _load_protocol(self, protocol_file: Path) -> List[Dict]:
        """Load protocol file and parse samples."""
        samples = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Format: SPEAKER FILE - ATTACK LABEL
                # Example: LA_0079 LA_E_5932896 - A17 spoof
                speaker_id = parts[0]
                file_id = parts[1]
                attack_type = parts[3]
                label = parts[4]  # 'bonafide' or 'spoof'
                
                samples.append({
                    'file_id': file_id,
                    'speaker_id': speaker_id,
                    'attack_type': attack_type,
                    'label': 0 if label == 'bonafide' else 1,  # 0=real, 1=fake
                    'label_str': label,
                    'dataset': 'asvspoof'
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Construct audio path
        audio_path = self.audio_dir / f"{sample_info['file_id']}.flac"
        
        # Process audio
        mel_spec = self.preprocessor.process_audio(audio_path, self.max_duration)
        
        # Apply additional transforms
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        # Return spectrogram, label, and metadata
        return {
            'spectrogram': mel_spec,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'file_id': sample_info['file_id'],
            'dataset': sample_info['dataset'],
            'attack_type': sample_info['attack_type']
        }


class WaveFakeDataset(Dataset):
    """
    WaveFake dataset.
    
    Dataset structure:
        WaveFake/
            generated_audio/
                <vocoder_name>/
                    *.wav
            real_audio/
                ljspeech/
                    *.wav
    """
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 subset: str = 'train',
                 preprocessor: Optional[AudioPreprocessor] = None,
                 max_duration: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 train_split: float = 0.8):
        """
        Initialize WaveFake dataset.
        
        Args:
            root_dir: Path to WaveFake directory
            subset: 'train' or 'test'
            preprocessor: AudioPreprocessor instance
            max_duration: Maximum audio duration
            transform: Additional transforms
            train_split: Train/test split ratio
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.max_duration = max_duration
        self.transform = transform
        
        # Collect all samples
        self.samples = self._collect_samples(train_split)
    
    def _collect_samples(self, train_split: float) -> List[Dict]:
        """Collect all audio files and create train/test split."""
        samples = []
        
        # Collect real audio
        real_dir = self.root_dir / 'real_audio' / 'ljspeech'
        if real_dir.exists():
            for audio_file in real_dir.glob('*.wav'):
                samples.append({
                    'path': audio_file,
                    'label': 0,  # real
                    'vocoder': 'real',
                    'dataset': 'wavefake'
                })
        
        # Collect fake audio from various vocoders
        fake_dir = self.root_dir / 'generated_audio'
        if fake_dir.exists():
            for vocoder_dir in fake_dir.iterdir():
                if vocoder_dir.is_dir():
                    vocoder_name = vocoder_dir.name
                    for audio_file in vocoder_dir.glob('*.wav'):
                        samples.append({
                            'path': audio_file,
                            'label': 1,  # fake
                            'vocoder': vocoder_name,
                            'dataset': 'wavefake'
                        })
        
        # Sort for reproducibility
        samples = sorted(samples, key=lambda x: str(x['path']))
        
        # Split train/test
        split_idx = int(len(samples) * train_split)
        if self.subset == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Process audio
        mel_spec = self.preprocessor.process_audio(sample_info['path'], self.max_duration)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return {
            'spectrogram': mel_spec,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'file_id': sample_info['path'].stem,
            'dataset': sample_info['dataset'],
            'attack_type': sample_info['vocoder']
        }


class FakeAVCelebDataset(Dataset):
    """
    FakeAVCeleb dataset (audio only).
    
    Dataset structure:
        FakeAVCeleb/
            RealVideo-RealAudio/
                *.mp4
            FakeVideo-FakeAudio/
                *.mp4
            RealVideo-FakeAudio/
                *.mp4
    """
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 subset: str = 'train',
                 preprocessor: Optional[AudioPreprocessor] = None,
                 max_duration: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 train_split: float = 0.8,
                 use_fake_audio_only: bool = True):
        """
        Initialize FakeAVCeleb dataset.
        
        Args:
            root_dir: Path to FakeAVCeleb directory
            subset: 'train' or 'test'
            preprocessor: AudioPreprocessor instance
            max_duration: Maximum audio duration
            transform: Additional transforms
            train_split: Train/test split ratio
            use_fake_audio_only: If True, only use samples with fake audio
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.max_duration = max_duration
        self.transform = transform
        self.use_fake_audio_only = use_fake_audio_only
        
        # Collect samples
        self.samples = self._collect_samples(train_split)
    
    def _collect_samples(self, train_split: float) -> List[Dict]:
        """Collect video files and extract audio."""
        samples = []
        
        # Real audio
        real_dir = self.root_dir / 'RealVideo-RealAudio'
        if real_dir.exists():
            for video_file in real_dir.glob('*.mp4'):
                samples.append({
                    'path': video_file,
                    'label': 0,  # real
                    'category': 'real-real',
                    'dataset': 'fakeavceleb'
                })
        
        # Fake audio (both categories)
        fake_dirs = [
            ('FakeVideo-FakeAudio', 'fake-fake'),
            ('RealVideo-FakeAudio', 'real-fake')
        ]
        
        for dir_name, category in fake_dirs:
            fake_dir = self.root_dir / dir_name
            if fake_dir.exists():
                for video_file in fake_dir.glob('*.mp4'):
                    samples.append({
                        'path': video_file,
                        'label': 1,  # fake
                        'category': category,
                        'dataset': 'fakeavceleb'
                    })
        
        # Sort for reproducibility
        samples = sorted(samples, key=lambda x: str(x['path']))
        
        # Split train/test
        split_idx = int(len(samples) * train_split)
        if self.subset == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Extract audio from video (will be done via torchaudio)
        # Note: torchaudio can read audio from mp4 files
        mel_spec = self.preprocessor.process_audio(sample_info['path'], self.max_duration)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return {
            'spectrogram': mel_spec,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'file_id': sample_info['path'].stem,
            'dataset': sample_info['dataset'],
            'attack_type': sample_info['category']
        }


class UnifiedDeepfakeDataset(Dataset):
    """
    Unified dataset combining multiple deepfake datasets.
    """
    
    def __init__(self,
                 datasets: List[Dataset],
                 balance: bool = True):
        """
        Initialize unified dataset.
        
        Args:
            datasets: List of dataset instances to combine
            balance: Whether to balance classes by undersampling
        """
        self.datasets = datasets
        self.combined = ConcatDataset(datasets)
        
        if balance:
            self._create_balanced_indices()
        else:
            self.indices = list(range(len(self.combined)))
    
    def _create_balanced_indices(self):
        """Create balanced dataset by undersampling majority class."""
        real_indices = []
        fake_indices = []
        
        for idx in range(len(self.combined)):
            sample = self.combined[idx]
            if sample['label'].item() == 0:
                real_indices.append(idx)
            else:
                fake_indices.append(idx)
        
        # Undersample to match minority class
        min_count = min(len(real_indices), len(fake_indices))
        
        import random
        random.shuffle(real_indices)
        random.shuffle(fake_indices)
        
        self.indices = real_indices[:min_count] + fake_indices[:min_count]
        random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.combined[self.indices[idx]]


def create_dataloaders(config: Dict,
                      preprocessor: Optional[AudioPreprocessor] = None) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for all datasets.
    
    Args:
        config: Configuration dictionary with dataset paths and settings
        preprocessor: AudioPreprocessor instance
        
    Returns:
        Dictionary of dataloaders
    """
    if preprocessor is None:
        preprocessor = AudioPreprocessor()
    
    datasets_to_load = []
    
    # ASVspoof
    if config.get('use_asvspoof', True):
        asvspoof_path = config.get('asvspoof_path')
        if asvspoof_path and Path(asvspoof_path).exists():
            for subset in ['train', 'dev']:
                ds = ASVspoofDataset(
                    root_dir=asvspoof_path,
                    subset=subset,
                    preprocessor=preprocessor,
                    max_duration=config.get('max_duration')
                )
                datasets_to_load.append(ds)
    
    # WaveFake
    if config.get('use_wavefake', True):
        wavefake_path = config.get('wavefake_path')
        if wavefake_path and Path(wavefake_path).exists():
            ds = WaveFakeDataset(
                root_dir=wavefake_path,
                subset='train',
                preprocessor=preprocessor,
                max_duration=config.get('max_duration')
            )
            datasets_to_load.append(ds)
    
    # FakeAVCeleb
    if config.get('use_fakeavceleb', True):
        fakeavceleb_path = config.get('fakeavceleb_path')
        if fakeavceleb_path and Path(fakeavceleb_path).exists():
            ds = FakeAVCelebDataset(
                root_dir=fakeavceleb_path,
                subset='train',
                preprocessor=preprocessor,
                max_duration=config.get('max_duration')
            )
            datasets_to_load.append(ds)
    
    # Combine datasets
    unified_train = UnifiedDeepfakeDataset(
        datasets=datasets_to_load,
        balance=config.get('balance_classes', True)
    )
    
    # Create test datasets similarly
    test_datasets = []
    
    if config.get('use_asvspoof', True):
        asvspoof_path = config.get('asvspoof_path')
        if asvspoof_path and Path(asvspoof_path).exists():
            test_datasets.append(ASVspoofDataset(
                root_dir=asvspoof_path,
                subset='eval',
                preprocessor=preprocessor,
                max_duration=config.get('max_duration')
            ))
    
    if config.get('use_wavefake', True):
        wavefake_path = config.get('wavefake_path')
        if wavefake_path and Path(wavefake_path).exists():
            test_datasets.append(WaveFakeDataset(
                root_dir=wavefake_path,
                subset='test',
                preprocessor=preprocessor,
                max_duration=config.get('max_duration')
            ))
    
    if config.get('use_fakeavceleb', True):
        fakeavceleb_path = config.get('fakeavceleb_path')
        if fakeavceleb_path and Path(fakeavceleb_path).exists():
            test_datasets.append(FakeAVCelebDataset(
                root_dir=fakeavceleb_path,
                subset='test',
                preprocessor=preprocessor,
                max_duration=config.get('max_duration')
            ))
    
    unified_test = UnifiedDeepfakeDataset(
        datasets=test_datasets,
        balance=False  # Don't balance test set
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        unified_train,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        unified_test,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }
