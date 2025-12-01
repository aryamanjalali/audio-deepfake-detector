"""
wav2vec2-based Model for Audio Deepfake Detection

Uses pretrained wav2vec2 embeddings with a classification head.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional


class Wav2Vec2Classifier(nn.Module):
    """
    wav2vec2-based deepfake classifier.
    
    Architecture:
    - Pretrained wav2vec2 encoder
    - Optional frozen/fine-tuned layers
    - Classification head
    """
    
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 num_classes: int = 2,
                 freeze_feature_extractor: bool = True,
                 freeze_encoder: bool = False,
                 num_frozen_layers: int = 0,
                 dropout: float = 0.3,
                 use_weighted_layer_sum: bool = False):
        """
        Initialize wav2vec2 classifier.
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            freeze_feature_extractor: Freeze CNN feature extractor
            freeze_encoder: Freeze entire transformer encoder
            num_frozen_layers: Number of encoder layers to freeze (if not freezing all)
            dropout: Dropout probability
            use_weighted_layer_sum: Use weighted sum of encoder layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_weighted_layer_sum = use_weighted_layer_sum
        
        # Load pretrained wav2vec2
        print(f"Loading pretrained model: {model_name}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=use_weighted_layer_sum
        )
        
        hidden_size = self.wav2vec2.config.hidden_size
        
        # Freeze feature extractor (CNN layers)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor.requires_grad_(False)
        
        # Freeze encoder layers
        if freeze_encoder:
            self.wav2vec2.encoder.requires_grad_(False)
        elif num_frozen_layers > 0:
            for layer in self.wav2vec2.encoder.layers[:num_frozen_layers]:
                layer.requires_grad_(False)
        
        # Weighted layer sum (if enabled)
        if use_weighted_layer_sum:
            num_layers = self.wav2vec2.config.num_hidden_layers + 1  # +1 for input
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_values, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_values: Raw audio waveform [batch, time]
            attention_mask: Attention mask [batch, time]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Ensure input is 1D per sample
        if input_values.dim() == 3:
            input_values = input_values.squeeze(1)  # Remove channel dimension
        
        # Get wav2vec2 embeddings
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layer_sum
        )
        
        # Extract features
        if self.use_weighted_layer_sum:
            # Weighted sum of all layers
            hidden_states = outputs.hidden_states
            stacked_hidden = torch.stack(hidden_states, dim=0)  # [num_layers, batch, time, hidden]
            weights = torch.softmax(self.layer_weights, dim=0)
            weighted_hidden = torch.sum(stacked_hidden * weights.view(-1, 1, 1, 1), dim=0)
            embeddings = weighted_hidden.mean(dim=1)  # Mean pool over time
        else:
            # Use final layer
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pool over time
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
    
    def predict(self, input_values, attention_mask=None):
        """
        Predict class probabilities.
        
        Args:
            input_values: Raw audio waveform
            attention_mask: Attention mask
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        logits = self.forward(input_values, attention_mask)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def freeze_base_model(self):
        """Freeze the entire wav2vec2 base model."""
        self.wav2vec2.requires_grad_(False)
    
    def unfreeze_base_model(self):
        """Unfreeze the entire wav2vec2 base model."""
        self.wav2vec2.requires_grad_(True)
    
    def unfreeze_top_layers(self, num_layers: int):
        """
        Unfreeze top N transformer layers for fine-tuning.
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        # Freeze all first
        self.wav2vec2.encoder.requires_grad_(False)
        
        # Unfreeze top layers
        for layer in self.wav2vec2.encoder.layers[-num_layers:]:
            layer.requires_grad_(True)


class Wav2Vec2ForRawAudio(nn.Module):
    """
    Simplified wrapper for wav2vec2 that handles raw audio input.
    
    This version expects raw audio waveforms (not spectrograms).
    """
    
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 num_classes: int = 2,
                 sample_rate: int = 16000):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        
        # Main classifier
        self.classifier = Wav2Vec2Classifier(
            model_name=model_name,
            num_classes=num_classes,
            freeze_feature_extractor=True,
            freeze_encoder=False,
            num_frozen_layers=6,  # Freeze first half of layers
            dropout=0.3
        )
    
    def forward(self, waveform):
        """
        Forward pass on raw audio.
        
        Args:
            waveform: Raw audio [batch, samples] or [batch, 1, samples]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Handle different input shapes
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [batch, samples]
        
        return self.classifier(waveform)
    
    def predict(self, waveform):
        """Predict on raw audio."""
        return self.classifier.predict(waveform)


def create_wav2vec2_model(config: dict) -> nn.Module:
    """
    Factory function to create wav2vec2 model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_name = config.get('wav2vec2_model', 'facebook/wav2vec2-base')
    
    model = Wav2Vec2Classifier(
        model_name=model_name,
        num_classes=config.get('num_classes', 2),
        freeze_feature_extractor=config.get('freeze_feature_extractor', True),
        freeze_encoder=config.get('freeze_encoder', False),
        num_frozen_layers=config.get('num_frozen_layers', 6),
        dropout=config.get('dropout', 0.3),
        use_weighted_layer_sum=config.get('use_weighted_layer_sum', False)
    )
    
    return model


if __name__ == '__main__':
    print("Testing wav2vec2 Classifier...")
    
    # Note: This will download the model if not cached
    print("Loading pretrained model (this may take a moment)...")
    
    try:
        model = Wav2Vec2Classifier(
            model_name="facebook/wav2vec2-base",
            num_classes=2,
            freeze_feature_extractor=True,
            num_frozen_layers=6
        )
        
        print(f"✓ Model loaded successfully")
        
        # Test with dummy audio (16kHz, 3 seconds)
        dummy_audio = torch.randn(2, 16000 * 3)
        
        output = model(dummy_audio)
        print(f"  Input shape: {dummy_audio.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test prediction
        preds, probs = model.predict(dummy_audio)
        print(f"  Predictions: {preds}")
        print(f"  Probabilities: {probs}")
        
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("  This is expected if you don't have internet connection")
        print("  The model will work when you have access to HuggingFace")
