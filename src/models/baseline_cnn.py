utf-8"""
Baseline CNN Model for Audio Deepfake Detection
ResNet-style CNN operating on log-mel spectrograms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out
class BaselineCNN(nn.Module):
    """
    Baseline CNN for deepfake detection.
    Architecture:
    - Input: Log-mel spectrogram [batch, 1, n_mels, time]
    - Conv stem
    - 4 ResNet blocks with increasing channels
    - Global average pooling
    - FC classifier
    """
    def __init__(self,
                 input_channels: int = 1,
                 n_mels: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        """
        Initialize baseline CNN.
        Args:
            input_channels: Number of input channels (1 for mono)
            n_mels: Number of mel bins
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input spectrogram [batch, channels, n_mels, time]
        Returns:
            Logits [batch, num_classes]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    def predict(self, x):
        """
        Predict class probabilities.
        Args:
            x: Input spectrogram
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs
class LightCNN(nn.Module):
    """
    Lighter CNN variant for faster training/inference.
    Good for initial experiments and rapid iteration.
    """
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
def create_baseline_model(config: dict) -> nn.Module:
    """
    Factory function to create baseline model.
    Args:
        config: Configuration dictionary
    Returns:
        Model instance
    """
    model_type = config.get('model_type', 'baseline')
    if model_type == 'baseline':
        model = BaselineCNN(
            input_channels=config.get('input_channels', 1),
            n_mels=config.get('n_mels', 128),
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == 'light':
        model = LightCNN(
            input_channels=config.get('input_channels', 1),
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model
if __name__ == '__main__':
    print("Testing Baseline CNN...")
    model = BaselineCNN(n_mels=128, num_classes=2)
    dummy_input = torch.randn(4, 1, 128, 256)
    output = model(dummy_input)
    print(f" Model initialized")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    preds, probs = model.predict(dummy_input)
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Probabilities shape: {probs.shape}")