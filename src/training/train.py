utf-8"""
Training Script for Audio Deepfake Detection Models
Supports training all three model architectures:
- Baseline CNN
- Augmented CNN (with codec augmentation)
- wav2vec2 (pretrained embeddings)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import create_dataloaders
from data.preprocessing import AudioPreprocessor
from models.baseline_cnn import BaselineCNN
from models.augmented_cnn import AugmentedCNN
from models.wav2vec2_model import Wav2Vec2Classifier
from utils.metrics import MetricsTracker, compute_metrics
from utils.audio_utils import count_parameters, get_device, save_checkpoint
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def create_model(config: dict, device: torch.device):
    """Create model based on configuration."""
    model_type = config['model']['type']
    if model_type == 'baseline_cnn':
        model = BaselineCNN(
            n_mels=config['audio']['n_mels'],
            num_classes=2
        )
    elif model_type == 'augmented_cnn':
        model = AugmentedCNN(
            n_mels=config['audio']['n_mels'],
            num_classes=2
        )
    elif model_type == 'wav2vec2':
        model = Wav2Vec2Classifier(
            num_classes=2,
            freeze_feature_extractor=config['model'].get('freeze_feature_extractor', True),
            freeze_encoder_layers=config['model'].get('freeze_encoder_layers', 6)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model = model.to(device)
    print(f"Created {model_type} model with {count_parameters(model):,} parameters")
    return model
def train_epoch(model, dataloader, criterion, optimizer, device, metrics_tracker, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, torch.softmax(
        torch.tensor([[1-p, p] for p in all_preds], dtype=torch.float32), dim=1
    ).numpy())
    metrics_tracker.update('train', epoch, avg_loss, metrics)
    return avg_loss, metrics
def validate_epoch(model, dataloader, criterion, device, metrics_tracker, epoch):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics_tracker.update('val', epoch, avg_loss, metrics)
    return avg_loss, metrics
def train(config_path: str):
    """Main training function."""
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    print(f"Training {config['model']['type']} model")
    device = get_device()
    print(f"Using device: {device}")
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    preprocessor = AudioPreprocessor(
        target_sample_rate=config['audio'].get('sample_rate', 16000),
        n_mels=config['audio'].get('n_mels', 128),
        n_fft=config['audio'].get('n_fft', 1024),
        hop_length=config['audio'].get('hop_length', 512)
    )
    print("Creating dataloaders...")
    try:
        dataloaders = create_dataloaders(config, preprocessor)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"\n Error creating dataloaders: {e}")
        print("\nThis usually means the datasets are not downloaded yet.")
        print("Please run: ./run.sh python scripts/download_datasets.py")
        return
    model = create_model(config, device)
    criterion = nn.CrossEntropyLoss()
    lr = config['training']['learning_rate']
    if config['model']['type'] == 'wav2vec2':
        optimizer = optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': lr * 0.1},  
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=config['training'].get('weight_decay', 0.01))
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
    scheduler_type = config['training'].get('scheduler', 'reduce_on_plateau')
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    metrics_tracker = MetricsTracker(save_dir=output_dir)
    num_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience', 10)
    patience_counter = 0
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics_tracker, epoch
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, metrics_tracker, epoch
        )
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics.get('roc_auc', 0):.4f}")
        if scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                output_dir / 'best_model.pth'
            )
            print(" Saved best model")
        else:
            patience_counter += 1
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                output_dir / f'checkpoint_epoch_{epoch}.pth'
            )
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    metrics_tracker.save_history()
    metrics_tracker.plot_training_curves(output_dir / 'training_curves.png')
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("=" * 60)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train audio deepfake detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    train(args.config)