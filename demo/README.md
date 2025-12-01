---
title: Audio Deepfake Detector
emoji: 🎙️
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Audio Deepfake Detection System

Robust detection of audio deepfakes under real-world compression scenarios.

## Features

- **3 Detection Models**: Baseline CNN, Augmented CNN, wav2vec2
- **Codec Simulation**: Test robustness under WhatsApp, Instagram, Phone, etc.
- **Real-time Inference**: Upload audio and get instant predictions
- **Visualization**: Confidence scores and spectrogram analysis

## Models

1. **Baseline CNN**: ResNet-style architecture on log-mel spectrograms
2. **Augmented CNN**: Same architecture, trained with codec augmentation
3. **wav2vec2**: Fine-tuned pretrained transformer

## Research

This system evaluates how audio deepfake detectors perform when audio undergoes real-world compression (social media sharing, messenger apps, etc.).

**Datasets**: ASVspoof 2019, WaveFake, FakeAVCeleb

**Paper**: [Coming soon]

## Usage

1. Upload an audio file or record using your microphone
2. Select a detection model
3. Optionally choose a codec simulation
4. Click "Detect Deepfake"

## Disclaimer

This is a research demonstration. Predictions are probabilistic and should not be used as definitive proof of authenticity.
