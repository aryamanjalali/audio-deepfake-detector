utf-8"""
Gradio Demo App for Audio Deepfake Detection
Interactive web interface for testing deepfake detection with codec simulation.
"""
import gradio as gr
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.codec_simulator import CodecSimulator
from data.preprocessing import AudioPreprocessor
from models.baseline_cnn import BaselineCNN
from models.augmented_cnn import AugmentedCNN
from models.wav2vec2_model import Wav2Vec2Classifier
import torchaudio
import matplotlib.pyplot as plt
class DeepfakeDetector:
    """Wrapper class for deepfake detection demo."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        self.codec_simulator = CodecSimulator()
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            n_mels=128
        )
        self.models = {}
        self.model_paths = {
            'baseline': 'experiments/results/baseline/best_model.pth',
            'augmented': 'experiments/results/augmented/best_model.pth',
            'wav2vec2': 'experiments/results/wav2vec2/best_model.pth'
        }
    def load_model(self, model_type: str):
        """Load model if not already loaded."""
        if model_type in self.models:
            return self.models[model_type]
        model_path = Path(self.model_paths[model_type])
        if model_type == 'baseline':
            model = BaselineCNN(n_mels=128, num_classes=2)
        elif model_type == 'augmented':
            model = AugmentedCNN(n_mels=128, num_classes=2)
        elif model_type == 'wav2vec2':
            model = Wav2Vec2Classifier(num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f" Loaded {model_type} model from {model_path}")
        else:
            print(f"⚠ Model weights not found at {model_path}. Using untrained model.")
        model.to(self.device)
        model.eval()
        self.models[model_type] = model
        return model
    def apply_codec_simulation(self, audio_path: str, simulation: str):
        """Apply codec simulation to audio file."""
        if simulation == "None (Original)":
            return audio_path, None
        simulation_map = {
            "WhatsApp Voice": "whatsapp",
            "Instagram Reels": "instagram",
            "Phone Call": "phone",
            "TikTok": "tiktok",
            "YouTube": "youtube"
        }
        if simulation in simulation_map:
            platform = simulation_map[simulation]
            compressed_path, metadata = self.codec_simulator.apply_social_media_chain(
                input_path=audio_path,
                platform=platform
            )
            return compressed_path, metadata
        elif "AAC" in simulation:
            bitrate = int(simulation.split()[1].replace('k', ''))
            compressed_path, metadata = self.codec_simulator.apply_codec(
                input_path=audio_path,
                codec='aac',
                bitrate=bitrate
            )
            return compressed_path, metadata
        elif "Opus" in simulation:
            bitrate = int(simulation.split()[1].replace('k', ''))
            compressed_path, metadata = self.codec_simulator.apply_codec(
                input_path=audio_path,
                codec='opus',
                bitrate=bitrate
            )
            return compressed_path, metadata
        elif "MP3" in simulation:
            bitrate = int(simulation.split()[1].replace('k', ''))
            compressed_path, metadata = self.codec_simulator.apply_codec(
                input_path=audio_path,
                codec='mp3',
                bitrate=bitrate
            )
            return compressed_path, metadata
        return audio_path, None
    def predict(self, audio_path: str, model_type: str, simulation: str):
        """
        Make prediction on audio file.
        Returns:
            Tuple of (prediction_text, confidence, spectrogram_plot, metadata_text)
        """
        try:
            processed_path, codec_metadata = self.apply_codec_simulation(audio_path, simulation)
            model = self.load_model(model_type)
            if model_type == 'wav2vec2':
                try:
                    waveform, sr = torchaudio.load(processed_path, backend="soundfile")
                except:
                    try:
                        waveform, sr = torchaudio.load(processed_path, backend="sox_io")
                    except:
                        waveform, sr = torchaudio.load(processed_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                max_samples = 16000 * 4
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                input_tensor = waveform.to(self.device)
            else:
                mel_spec = self.preprocessor.process_audio(processed_path, max_duration=4.0)
                input_tensor = mel_spec.unsqueeze(0).to(self.device)  
            with torch.no_grad():
                preds, probs = model.predict(input_tensor)
            pred_class = preds[0].item()
            confidence = probs[0].cpu().numpy()
            fake_prob = confidence[1] * 100  
            real_prob = confidence[0] * 100  
            if pred_class == 0:
                prediction_text = f"🟢 **REAL AUDIO** (Confidence: {real_prob:.1f}%)"
            else:
                prediction_text = f" **FAKE AUDIO** (Confidence: {fake_prob:.1f}%)"
            fig, ax = plt.subplots(figsize=(8, 3))
            categories = ['Real', 'Fake']
            values = [real_prob, fake_prob]
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.barh(categories, values, color=colors, alpha=0.7)
            ax.set_xlim([0, 100])
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Deepfake Detection Confidence')
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 2, i, f'{val:.1f}%', va='center')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            spec_fig = None
            if model_type != 'wav2vec2':
                spec_fig, spec_ax = plt.subplots(figsize=(10, 4))
                spec_ax.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
                spec_ax.set_xlabel('Time')
                spec_ax.set_ylabel('Mel Frequency Bin')
                spec_ax.set_title('Log-Mel Spectrogram')
                plt.colorbar(spec_ax.images[0], ax=spec_ax, label='Magnitude (dB)')
                plt.tight_layout()
            metadata_text = f"**Model Used:** {model_type.capitalize()}\n"
            metadata_text += f"**Simulation:** {simulation}\n"
            if codec_metadata:
                metadata_text += f"\n**Codec Details:**\n"
                for key, value in codec_metadata.items():
                    if key not in ['input_path', 'output_path']:
                        metadata_text += f"- {key}: {value}\n"
            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except:
                    pass
            return prediction_text, fig, spec_fig, metadata_text
        except Exception as e:
            return f" Error: {str(e)}", None, None, ""
detector = DeepfakeDetector()
def create_demo():
    """Create Gradio demo interface."""
    with gr.Blocks(title="Audio Deepfake Detector") as demo:
        gr.Markdown("# ️ Audio Deepfake Detection System")
        gr.Markdown(
            "Upload an audio file to detect if it's real or AI-generated. "
            "Optionally simulate real-world compression (WhatsApp, Instagram, etc.) "
            "to test detection robustness."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                model_choice = gr.Radio(
                    choices=["baseline", "augmented", "wav2vec2"],
                    value="baseline",
                    label="Detection Model",
                    info="baseline: Spectrogram CNN | augmented: Robust CNN | wav2vec2: Pretrained Embeddings"
                )
                simulation_choice = gr.Dropdown(
                    choices=[
                        "None (Original)",
                        "WhatsApp Voice",
                        "Instagram Reels",
                        "Phone Call",
                        "TikTok",
                        "YouTube",
                        "AAC 128k",
                        "AAC 64k",
                        "AAC 32k",
                        "Opus 64k",
                        "Opus 32k",
                        "MP3 128k",
                        "MP3 64k"
                    ],
                    value="None (Original)",
                    label="Codec Simulation",
                    info="Simulate real-world compression before detection"
                )
                detect_btn = gr.Button(" Detect Deepfake", variant="primary", size="lg")
            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction")
                confidence_plot = gr.Plot(label="Confidence Scores")
                metadata_output = gr.Markdown(label="Details")
        with gr.Row():
            spectrogram_plot = gr.Plot(label="Spectrogram Analysis")
        gr.Markdown("## ℹ️ How it works")
        gr.Markdown(
            "1. **Baseline Model**: CNN trained on clean spectrograms\n"
            "2. **Augmented Model**: CNN trained with codec augmentation for robustness\n"
            "3. **wav2vec2 Model**: Fine-tuned pretrained transformer for transfer learning\n\n"
            "The system can simulate various compression scenarios to test how well "
            "detection works 'in the wild' after audio is shared through social media or messenger apps."
        )
        gr.Markdown(
            "⚠️ **Disclaimer**: This is a research demonstration. "
            "Predictions are probabilistic and should not be used as definitive proof. "
            "Always verify audio authenticity through multiple methods."
        )
        detect_btn.click(
            fn=detector.predict,
            inputs=[audio_input, model_choice, simulation_choice],
            outputs=[prediction_output, confidence_plot, spectrogram_plot, metadata_output]
        )
    return demo
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  
    )