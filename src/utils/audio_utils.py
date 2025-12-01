utf-8"""
Utility functions for audio processing and model helpers.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Optional
def load_audio_file(path: Union[str, Path],
                   target_sr: int = 16000,
                   max_duration: Optional[float] = None) -> torch.Tensor:
    """
    Load and preprocess audio file.
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        max_duration: Maximum duration in seconds
    Returns:
        Waveform tensor [1, samples]
    """
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if max_duration:
        max_samples = int(max_duration * target_sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    return waveform
def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.
    Args:
        model: PyTorch model
    Returns:
        Dict with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get best available device.
    Args:
        prefer_cuda: Prefer CUDA if available
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   metrics: dict,
                   path: Union[str, Path]):
    """
    Save training checkpoint.
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
def load_checkpoint(path: Union[str, Path],
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Load training checkpoint.
    Args:
        path: Checkpoint path
        model: Model instance
        optimizer: Optimizer instance (optional)
    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint