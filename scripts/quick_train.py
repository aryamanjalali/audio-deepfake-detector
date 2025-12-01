"""
Simplified Training Script for Quick Demo

Trains on a simple directory structure with real/fake folders.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import AudioPreprocessor
from models.baseline_cnn import BaselineCNN
from utils.audio_utils import save_checkpoint, get_device


class SimpleAudioDataset(Dataset):
    """Simple dataset from real/fake folders."""
    
    def __init__(self, real_dir, fake_dir, preprocessor, max_duration=4.0):
        self.preprocessor = preprocessor
        self.max_duration = max_duration
        
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
        print(f"  Real: {sum(1 for _, label in self.files if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.files if label == 1)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path, label = self.files[idx]
        
        try:
            # Process audio
            mel_spec = self.preprocessor.process_audio(path, max_duration=self.max_duration)
            return mel_spec, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zeros if file fails
            return torch.zeros(1, 128, 256), label


def train_quick():
    """Quick training function."""
    print("="*60)
    print("Quick Demo Training")
    print("="*60)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Paths
    real_dir = "data/raw/quick_train/real"
    fake_dir = "data/raw/quick_train/fake"
    output_dir = Path("experiments/results/quick_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check directories exist
    if not Path(real_dir).exists() or not Path(fake_dir).exists():
        print("\n❌ Dataset directories not found!")
        print("Run this first: ./run.sh python scripts/setup_quick_train.py")
        return
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=128
    )
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = SimpleAudioDataset(real_dir, fake_dir, preprocessor)
    
    if len(full_dataset) < 4:
        print("\n❌ Need at least 4 audio files to train!")
        print("Add more files to:")
        print(f"  {real_dir}")
        print(f"  {fake_dir}")
        return
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = BaselineCNN(n_mels=128, num_classes=2).to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    num_epochs = 10
    best_val_acc = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, {'accuracy': val_acc},
                output_dir / 'best_model.pth'
            )
            print(f"  ✓ Saved best model (acc={val_acc:.3f})")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("\nNext steps:")
    print("1. Restart demo: pkill -f 'demo/app.py' && ./run.sh python demo/app.py")
    print("2. Test on your audio file in the web interface!")
    print("="*60)


if __name__ == '__main__':
    train_quick()
