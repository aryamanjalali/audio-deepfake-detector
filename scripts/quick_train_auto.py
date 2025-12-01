"""
Quick Auto-Train Script

Trains on auto-downloaded LJSpeech + synthetic fake dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import random
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import AudioPreprocessor  
from models.baseline_cnn import BaselineCNN
from utils.audio_utils import save_checkpoint, get_device


class SimpleAudioDataset(Dataset):
    """Simple dataset from real/fake folders."""
    
    def __init__(self, real_dir, fake_dir, preprocessor, max_duration=4.0, return_raw=False):
        self.preprocessor = preprocessor
        self.max_duration = max_duration
        self.return_raw = return_raw
        
        # Find all audio files
        audio_exts = ['*.wav', '*.mp3', '*.opus', '*.m4a', '*.flac']
        self.files = []
        
        real_path = Path(real_dir)
        fake_path = Path(fake_dir)
        
        # Real files (label 0)
        for ext in audio_exts:
            for f in real_path.glob(ext):
                self.files.append((str(f), 0))
        
        # Fake files (label 1) 
        for ext in audio_exts:
            for f in fake_path.glob(ext):
                self.files.append((str(f), 1))
        
        # Shuffle
        random.shuffle(self.files)
        
        print(f"Found {len(self.files)} audio files")
        real_count = sum(1 for _, label in self.files if label == 0)
        fake_count = sum(1 for _, label in self.files if label == 1)
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path, label = self.files[idx]
        
        try:
            if self.return_raw:
                # Load raw audio for wav2vec2
                waveform, sr = self.preprocessor.load_audio(path, max_duration=self.max_duration)
                # Ensure 16kHz
                if sr != 16000:
                    import torchaudio.transforms as T
                    resampler = T.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Squeeze channel dim for 1D output if needed, but usually we keep [1, time] or [time]
                # Wav2Vec2 expects [batch, time]
                return waveform.squeeze(0), label
            else:
                # Process audio to spectrogram
                mel_spec = self.preprocessor.process_audio(path, max_duration=self.max_duration)
                return mel_spec, label
        except Exception as e:
            # Return zeros if file fails to load
            if self.return_raw:
                return torch.zeros(16000 * 4), label # Approx 4s of silence
            else:
                return torch.zeros(1, 128, 256), label


import argparse
from models.augmented_cnn import AugmentedCNN
from models.wav2vec2_model import Wav2Vec2Classifier

def train_auto(model_type='baseline'):
    """Training function for auto-downloaded dataset."""
    print("="*60)
    print(f"Training Audio Deepfake Detector ({model_type})")
    print("="*60)
    
    # Load config
    config_file = Path('experiments/configs/auto_train.json')
    if not config_file.exists():
        print("\n❌ Config file not found!")
        print("Run this first: ./run.sh python scripts/download_auto_dataset.py")
        return
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(f"experiments/results/auto_trained_{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=128
    )
    
    # Determine if we need raw audio
    return_raw = (model_type == 'wav2vec2')
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = SimpleAudioDataset(
        config['real_dir'], 
        config['fake_dir'],
        preprocessor,
        return_raw=return_raw
    )
    
    if len(full_dataset) < 10:
        print("\n❌ Dataset too small!")
        print("Need at least 10 samples. Did the download complete?")
        return
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Custom collate function to pad tensors to same size
    def collate_pad(batch):
        """Pad tensors to max length in batch."""
        data, labels = zip(*batch)
        
        # Check if 1D (raw) or 2D/3D (spectrogram)
        is_raw = (data[0].dim() == 1)
        
        # Find max time dimension
        max_time = max(d.shape[-1] for d in data)
        
        # Pad all to max_time
        padded_data = []
        for d in data:
            if d.shape[-1] < max_time:
                padding = max_time - d.shape[-1]
                if is_raw:
                    padded = torch.nn.functional.pad(d, (0, padding))
                else:
                    padded = torch.nn.functional.pad(d, (0, padding))
                padded_data.append(padded)
            else:
                padded_data.append(d)
        
        return torch.stack(padded_data), torch.tensor(labels)
    
    # Wav2Vec2 uses raw audio, not spectrograms, so we need a different collate if we were doing full pipeline
    # But here SimpleAudioDataset returns spectrograms. 
    # Wait, Wav2Vec2 needs raw audio.
    # I need to check SimpleAudioDataset. It calls preprocessor.process_audio which returns mel spec.
    # This script needs modification to handle Wav2Vec2 which expects raw audio.
    
    # Let's adjust SimpleAudioDataset to return raw audio if needed, or handle it here.
    # Actually, let's check the preprocessor.
    
    batch_size = config.get('batch_size', 8)
    
    # For Wav2Vec2, we might need smaller batch size on GPU/MPS
    if model_type == 'wav2vec2':
        batch_size = max(1, batch_size // 2)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_pad)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    if model_type == 'baseline':
        model = BaselineCNN(n_mels=128, num_classes=2)
    elif model_type == 'augmented':
        model = AugmentedCNN(n_mels=128, num_classes=2)
    elif model_type == 'wav2vec2':
        model = Wav2Vec2Classifier(num_classes=2, freeze_feature_extractor=True)
        
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created {model_type} model with {num_params:,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Train
    num_epochs = config.get('epochs', 20)
    best_val_acc = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # For Wav2Vec2 in this specific script, we might have issues if we pass spectrograms
            # The Wav2Vec2Classifier in src/models/wav2vec2_model.py expects raw waveforms?
            # Let's check the model definition.
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, {'accuracy': val_acc},
                output_dir / 'best_model.pth'
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'augmented', 'wav2vec2'])
    args = parser.parse_args()
    
    train_auto(args.model)
