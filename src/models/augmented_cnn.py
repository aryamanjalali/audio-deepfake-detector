utf-8"""
Data-Augmented CNN Model
Same architecture as baseline but with on-the-fly codec augmentation during training.
"""
import torch
import torch.nn as nn
from .baseline_cnn import BaselineCNN, LightCNN
import random
import tempfile
import os
from pathlib import Path
class CodecAugmentation:
    """
    On-the-fly codec augmentation for training.
    Applies random codec transformations to spectrograms during training.
    """
    def __init__(self,
                 codec_simulator=None,
                 preprocessor=None,
                 augmentation_prob: float = 0.5,
                 codecs: list = None,
                 bitrates: dict = None):
        """
        Initialize codec augmentation.
        Args:
            codec_simulator: CodecSimulator instance
            preprocessor: AudioPreprocessor instance
            augmentation_prob: Probability of applying augmentation
            codecs: List of codecs to use
            bitrates: Dict of codec -> list of bitrates
        """
        self.codec_simulator = codec_simulator
        self.preprocessor = preprocessor
        self.augmentation_prob = augmentation_prob
        self.codecs = codecs or ['aac', 'opus', 'mp3']
        self.bitrates = bitrates or {
            'aac': [128, 64, 32],
            'opus': [64, 32, 24],
            'mp3': [128, 64, 32]
        }
    def __call__(self, spectrogram, audio_path=None):
        """
        Apply random codec augmentation.
        Args:
            spectrogram: Input spectrogram (not used if audio_path provided)
            audio_path: Path to original audio file
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.augmentation_prob:
            return spectrogram
        if audio_path and self.codec_simulator and self.preprocessor:
            try:
                codec = random.choice(self.codecs)
                bitrate = random.choice(self.bitrates[codec])
                compressed_path, _ = self.codec_simulator.apply_codec(
                    input_path=str(audio_path),
                    codec=codec,
                    bitrate=bitrate
                )
                augmented_spec = self.preprocessor.process_audio(compressed_path)
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
                return augmented_spec
            except Exception as e:
                return spectrogram
        return self._apply_spec_augmentation(spectrogram)
    def _apply_spec_augmentation(self, spectrogram):
        """
        Apply spectrogram-level augmentation to simulate codec effects.
        This is a simplified version that doesn't require re-encoding.
        """
        if random.random() < 0.3:
            mask_freq = random.randint(1, 15)
            spectrogram[:, -mask_freq:, :] *= 0.1
        if random.random() < 0.3:
            mask_time = random.randint(1, 20)
            start_t = random.randint(0, max(0, spectrogram.shape[2] - mask_time))
            spectrogram[:, :, start_t:start_t+mask_time] *= 0.1
        if random.random() < 0.3:
            noise = torch.randn_like(spectrogram) * 0.01
            spectrogram = spectrogram + noise
        return spectrogram
class AugmentedCNN(BaselineCNN):
    """
    Augmented CNN that uses the same architecture as baseline
    but is trained with codec augmentation.
    This is primarily a marker class - the augmentation happens
    in the training loop via the CodecAugmentation transform.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
def create_augmented_model(config: dict) -> nn.Module:
    """
    Factory function to create augmented model.
    Args:
        config: Configuration dictionary
    Returns:
        Model instance (same architecture as baseline)
    """
    model = AugmentedCNN(
        input_channels=config.get('input_channels', 1),
        n_mels=config.get('n_mels', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3)
    )
    return model
if __name__ == '__main__':
    print("Testing Augmented CNN...")
    model = AugmentedCNN(n_mels=128, num_classes=2)
    dummy_input = torch.randn(4, 1, 128, 256)
    output = model(dummy_input)
    print(f" Augmented model initialized")
    print(f"  Architecture: Same as baseline")
    print(f"  Training: Uses codec augmentation")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n Testing spectrogram-level augmentation...")
    aug = CodecAugmentation(augmentation_prob=1.0)
    test_spec = torch.randn(1, 128, 256)
    aug_spec = aug(test_spec)
    print(f"  Original shape: {test_spec.shape}")
    print(f"  Augmented shape: {aug_spec.shape}")