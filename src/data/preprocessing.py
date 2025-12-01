"""
Audio Preprocessing Utilities

Handles audio loading, normalization, and feature extraction.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings


class AudioPreprocessor:
    """Preprocesses audio files for deepfake detection."""
    
    def __init__(self,
                 target_sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 win_length: Optional[int] = None,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 normalize: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            target_sample_rate: Target sample rate for all audio
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length (defaults to n_fft)
            f_min: Minimum frequency
            f_max: Maximum frequency (defaults to sr/2)
            normalize: Whether to normalize spectrograms
        """
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or target_sample_rate / 2
        self.normalize = normalize
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=2.0
        )
        
    def load_audio(self, 
                   path: Union[str, Path],
                   max_duration: Optional[float] = None) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to target sample rate.
        
        Args:
            path: Path to audio file
            max_duration: Maximum duration in seconds (optional)
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        # Load audio - try different backends to avoid torchcodec dependency
        try:
            waveform, sample_rate = torchaudio.load(str(path), backend="soundfile")
        except:
            try:
                waveform, sample_rate = torchaudio.load(str(path), backend="sox_io")
            except:
                # Fallback to default (may fail if torchcodec not installed)
                waveform, sample_rate = torchaudio.load(str(path))
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim to max duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
        
        return waveform, sample_rate
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio waveform to [-1, 1] range.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Normalized waveform
        """
        # Peak normalization
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def extract_mel_spectrogram(self,
                                waveform: torch.Tensor,
                                log_scale: bool = True) -> torch.Tensor:
        """
        Extract mel spectrogram from waveform.
        
        Args:
            waveform: Input waveform [channels, samples]
            log_scale: Apply log scaling
            
        Returns:
            Mel spectrogram [channels, n_mels, time]
        """
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Apply log scaling
        if log_scale:
            mel_spec = torch.log(mel_spec + 1e-9)  # Add epsilon to avoid log(0)
        
        # Normalize
        if self.normalize:
            mel_spec = self._normalize_spectrogram(mel_spec)
        
        return mel_spec
    
    def _normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram to zero mean and unit variance.
        
        Args:
            spec: Input spectrogram
            
        Returns:
            Normalized spectrogram
        """
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / std
        return spec
    
    def process_audio(self,
                     path: Union[str, Path],
                     max_duration: Optional[float] = None,
                     return_waveform: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load audio and extract mel spectrogram (one-step processing).
        
        Args:
            path: Path to audio file
            max_duration: Maximum duration in seconds
            return_waveform: Whether to also return the waveform
            
        Returns:
            Mel spectrogram, or (mel_spec, waveform) if return_waveform=True
        """
        # Load and normalize audio
        waveform, _ = self.load_audio(path, max_duration)
        waveform = self.normalize_audio(waveform)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(waveform)
        
        if return_waveform:
            return mel_spec, waveform
        return mel_spec
    
    def chunk_audio(self,
                   waveform: torch.Tensor,
                   chunk_duration: float,
                   overlap: float = 0.0) -> list[torch.Tensor]:
        """
        Split audio into overlapping chunks.
        
        Args:
            waveform: Input waveform [channels, samples]
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap ratio (0.0 to 1.0)
            
        Returns:
            List of waveform chunks
        """
        chunk_samples = int(chunk_duration * self.target_sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        chunks = []
        start = 0
        
        while start + chunk_samples <= waveform.shape[1]:
            chunk = waveform[:, start:start + chunk_samples]
            chunks.append(chunk)
            start += hop_samples
        
        # Handle last chunk if it's too short
        if start < waveform.shape[1] and len(chunks) > 0:
            # Pad last chunk to required length
            remaining = waveform[:, start:]
            padding = chunk_samples - remaining.shape[1]
            if padding > 0:
                padded = torch.nn.functional.pad(remaining, (0, padding))
                chunks.append(padded)
        
        return chunks


class DefensePipeline(AudioPreprocessor):
    """
    Enhanced preprocessor with defense mechanisms against codec artifacts.
    
    Applies DSP techniques to normalize away compression artifacts.
    """
    
    def __init__(self,
                 target_sample_rate: int = 16000,
                 bandpass_low: float = 150.0,
                 bandpass_high: float = 6000.0,
                 apply_denoise: bool = False,
                 **kwargs):
        """
        Initialize defense pipeline.
        
        Args:
            target_sample_rate: Target sample rate
            bandpass_low: Low cutoff for bandpass filter (Hz)
            bandpass_high: High cutoff for bandpass filter (Hz)
            apply_denoise: Whether to apply spectral gating
            **kwargs: Additional args for AudioPreprocessor
        """
        super().__init__(target_sample_rate=target_sample_rate, **kwargs)
        
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.apply_denoise = apply_denoise
        
    def apply_bandpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply bandpass filter to focus on speech frequencies.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Filtered waveform
        """
        # Create bandpass filter
        lowpass = torchaudio.transforms.Lowpass(
            self.target_sample_rate,
            cutoff_freq=self.bandpass_high
        )
        highpass = torchaudio.transforms.Highpass(
            self.target_sample_rate,
            cutoff_freq=self.bandpass_low
        )
        
        # Apply filters
        waveform = lowpass(waveform)
        waveform = highpass(waveform)
        
        return waveform
    
    def apply_spectral_gating(self,
                              waveform: torch.Tensor,
                              threshold_db: float = -40.0) -> torch.Tensor:
        """
        Simple spectral gating for noise reduction.
        
        Args:
            waveform: Input waveform
            threshold_db: Threshold in dB below which to gate
            
        Returns:
            Denoised waveform
        """
        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        
        # Compute magnitude
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold_db / 20)
        max_mag = magnitude.max()
        threshold = max_mag * threshold_linear
        
        # Apply gate
        mask = magnitude > threshold
        magnitude_gated = magnitude * mask
        
        # Reconstruct
        stft_gated = magnitude_gated * torch.exp(1j * phase)
        waveform_denoise = torch.istft(
            stft_gated,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=waveform.shape[1]
        ).unsqueeze(0)
        
        return waveform_denoise
    
    def process_audio(self,
                     path: Union[str, Path],
                     max_duration: Optional[float] = None,
                     return_waveform: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process audio with defense pipeline.
        
        Args:
            path: Path to audio file
            max_duration: Maximum duration
            return_waveform: Whether to return waveform
            
        Returns:
            Processed mel spectrogram (and optionally waveform)
        """
        # Load audio
        waveform, _ = self.load_audio(path, max_duration)
        
        # Apply defense mechanisms
        waveform = self.normalize_audio(waveform)
        waveform = self.apply_bandpass(waveform)
        
        if self.apply_denoise:
            waveform = self.apply_spectral_gating(waveform)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(waveform)
        
        if return_waveform:
            return mel_spec, waveform
        return mel_spec


if __name__ == '__main__':
    # Example usage
    print("Audio Preprocessor - Testing...")
    
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=128
    )
    print(f"✓ Preprocessor initialized")
    print(f"  Sample rate: {preprocessor.target_sample_rate} Hz")
    print(f"  Mel bins: {preprocessor.n_mels}")
    print(f"  FFT size: {preprocessor.n_fft}")
    
    defense = DefensePipeline(
        target_sample_rate=16000,
        bandpass_low=150,
        bandpass_high=6000
    )
    print(f"\n✓ Defense pipeline initialized")
    print(f"  Bandpass: {defense.bandpass_low}-{defense.bandpass_high} Hz")
