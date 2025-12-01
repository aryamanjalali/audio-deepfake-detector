"""
Codec Simulator - ffmpeg Pipeline for Real-World Compression

Simulates various compression scenarios that audio experiences in the wild:
- Social media (WhatsApp, Instagram, TikTok)
- Phone calls
- Various codecs (AAC, Opus, MP3) at different bitrates
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
import shutil


class CodecSimulator:
    """Simulates real-world audio compression using ffmpeg."""
    
    # Codec configurations
    CODECS = {
        'aac': {
            'codec': 'aac',
            'extension': 'm4a',
            'bitrates': [128, 64, 32, 16]  # kbps
        },
        'opus': {
            'codec': 'libopus',
            'extension': 'opus',
            'bitrates': [64, 32, 24, 16]
        },
        'mp3': {
            'codec': 'libmp3lame',
            'extension': 'mp3',
            'bitrates': [128, 64, 32, 16]
        }
    }
    
    # Sample rates for testing
    SAMPLE_RATES = [48000, 16000, 8000]
    
    # Predefined social media chains
    SOCIAL_MEDIA_CHAINS = {
        'whatsapp': {
            'description': 'WhatsApp voice message',
            'sample_rate': 16000,
            'channels': 1,
            'codec': 'libopus',
            'bitrate': 24,
            'extension': 'opus'
        },
        'instagram': {
            'description': 'Instagram Reels audio',
            'sample_rate': 44100,
            'channels': 2,
            'codec': 'aac',
            'bitrate': 128,
            'extension': 'm4a',
            'chain': True,  # Re-encode after first encoding
            'second_bitrate': 64
        },
        'phone': {
            'description': 'Phone call quality',
            'sample_rate': 8000,
            'channels': 1,
            'codec': 'aac',
            'bitrate': 16,
            'extension': 'm4a',
            'lowpass': 4000  # Hz
        },
        'tiktok': {
            'description': 'TikTok audio',
            'sample_rate': 44100,
            'channels': 2,
            'codec': 'aac',
            'bitrate': 96,
            'extension': 'm4a'
        },
        'youtube': {
            'description': 'YouTube upload (simulated)',
            'sample_rate': 48000,
            'channels': 2,
            'codec': 'libopus',
            'bitrate': 128,
            'extension': 'webm'
        }
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize codec simulator.
        
        Args:
            output_dir: Directory to save transformed audio files
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Verify ffmpeg is installed."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt-get install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            )
    
    def apply_codec(self,
                   input_path: str,
                   codec: str,
                   bitrate: int,
                   sample_rate: Optional[int] = None,
                   channels: Optional[int] = None,
                   output_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Apply a specific codec transformation.
        
        Args:
            input_path: Path to input audio file
            codec: Codec name ('aac', 'opus', 'mp3')
            bitrate: Bitrate in kbps
            sample_rate: Target sample rate (optional)
            channels: Number of channels (optional)
            output_path: Where to save output (optional, will use temp if None)
            
        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if codec not in self.CODECS:
            raise ValueError(f"Unknown codec: {codec}. Choose from {list(self.CODECS.keys())}")
        
        codec_info = self.CODECS[codec]
        
        # Create output path if not provided
        if output_path is None:
            if self.output_dir:
                input_name = Path(input_path).stem
                output_path = self.output_dir / f"{input_name}_{codec}_{bitrate}k.{codec_info['extension']}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Use temporary file
                fd, output_path = tempfile.mkstemp(suffix=f'.{codec_info["extension"]}')
                os.close(fd)
        
        output_path = str(output_path)
        
        # Build ffmpeg command with -nostdin to prevent hanging
        cmd = ['ffmpeg', '-nostdin', '-y', '-i', input_path]
        
        # Sample rate
        if sample_rate:
            cmd.extend(['-ar', str(sample_rate)])
        
        # Channels
        if channels:
            cmd.extend(['-ac', str(channels)])
        
        # Codec and bitrate
        cmd.extend(['-c:a', codec_info['codec']])
        cmd.extend(['-b:a', f'{bitrate}k'])
        
        # Add faster encoding preset for speed
        if codec_info['codec'] == 'aac':
            cmd.extend(['-q:a', '2'])  # Use VBR quality mode (faster)
        elif codec_info['codec'] == 'libopus':
            cmd.extend(['-vbr', 'on', '-compression_level', '5'])  # Faster compression
        elif codec_info['codec'] == 'libmp3lame':
            cmd.extend(['-qscale:a', '4'])  # Use quality mode (faster)
        
        cmd.append(output_path)
        
        # Run ffmpeg with suppressed output for speed
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # Suppress verbose output
                check=True,
                timeout=30  # 30 second timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timed out after 30 seconds")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")
        
        # Create metadata
        metadata = {
            'codec': codec,
            'bitrate_kbps': bitrate,
            'sample_rate': sample_rate,
            'channels': channels,
            'transformation': f'{codec}_{bitrate}k',
            'input_path': input_path,
            'output_path': output_path
        }
        
        return output_path, metadata
    
    def apply_social_media_chain(self,
                                 input_path: str,
                                 platform: str,
                                 output_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Apply a predefined social media compression chain.
        
        Args:
            input_path: Path to input audio
            platform: Platform name (e.g., 'whatsapp', 'instagram')
            output_path: Where to save output (optional)
            
        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if platform not in self.SOCIAL_MEDIA_CHAINS:
            raise ValueError(
                f"Unknown platform: {platform}. "
                f"Choose from {list(self.SOCIAL_MEDIA_CHAINS.keys())}"
            )
        
        config = self.SOCIAL_MEDIA_CHAINS[platform]
        
        # Create output path
        if output_path is None:
            if self.output_dir:
                input_name = Path(input_path).stem
                output_path = self.output_dir / f"{input_name}_{platform}.{config['extension']}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                fd, output_path = tempfile.mkstemp(suffix=f'.{config["extension"]}')
                os.close(fd)
        
        output_path = str(output_path)
        
        # Build ffmpeg command with -nostdin to prevent hanging
        cmd = ['ffmpeg', '-nostdin', '-y', '-i', input_path]
        cmd.extend(['-ar', str(config['sample_rate'])])
        cmd.extend(['-ac', str(config['channels'])])
        cmd.extend(['-c:a', config['codec']])
        cmd.extend(['-b:a', f'{config["bitrate"]}k'])
        
        # Add lowpass filter for phone if specified
        if 'lowpass' in config:
            cmd.extend(['-af', f'lowpass=f={config["lowpass"]}'])
        
        # For Instagram, do a two-stage encode
        if config.get('chain'):
            # First encode
            temp_path = output_path + '.temp' + Path(output_path).suffix
            cmd.append(temp_path)
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, check=True, timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"ffmpeg timed out (stage 1)")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg failed (stage 1)")
            
            # Second encode (re-encode at lower bitrate)
            cmd = ['ffmpeg', '-nostdin', '-y', '-i', temp_path]
            cmd.extend(['-c:a', config['codec']])
            cmd.extend(['-b:a', f'{config["second_bitrate"]}k'])
            cmd.append(output_path)
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, check=True, timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"ffmpeg timed out (stage 2)")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg failed (stage 2)")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Single-stage encode
            cmd.append(output_path)
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, check=True, timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"ffmpeg timed out")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg failed: {e.stderr}")
        
        # Create metadata
        metadata = {
            'platform': platform,
            'description': config['description'],
            'codec': config['codec'],
            'bitrate_kbps': config['bitrate'],
            'sample_rate': config['sample_rate'],
            'channels': config['channels'],
            'transformation': f'{platform}_chain',
            'input_path': input_path,
            'output_path': output_path
        }
        
        if config.get('chain'):
            metadata['multi_stage'] = True
            metadata['second_bitrate_kbps'] = config['second_bitrate']
        
        if 'lowpass' in config:
            metadata['lowpass_hz'] = config['lowpass']
        
        return output_path, metadata
    
    def generate_codec_ladder(self,
                             input_path: str,
                             codec: str,
                             output_dir: Optional[Path] = None) -> Dict[int, Tuple[str, Dict]]:
        """
        Generate multiple versions of audio at different bitrates.
        
        Args:
            input_path: Path to input audio
            codec: Codec to use
            output_dir: Where to save outputs
            
        Returns:
            Dict mapping bitrate -> (output_path, metadata)
        """
        if codec not in self.CODECS:
            raise ValueError(f"Unknown codec: {codec}")
        
        bitrates = self.CODECS[codec]['bitrates']
        results = {}
        
        save_dir = output_dir or self.output_dir
        
        for bitrate in bitrates:
            output_path, metadata = self.apply_codec(
                input_path=input_path,
                codec=codec,
                bitrate=bitrate,
                output_path=None if not save_dir else 
                          str(save_dir / f"{Path(input_path).stem}_{codec}_{bitrate}k.{self.CODECS[codec]['extension']}")
            )
            results[bitrate] = (output_path, metadata)
        
        return results
    
    def generate_all_variants(self,
                             input_path: str,
                             output_dir: Optional[Path] = None,
                             include_social_media: bool = True) -> Dict[str, Tuple[str, Dict]]:
        """
        Generate all compression variants for comprehensive testing.
        
        Args:
            input_path: Path to input audio
            output_dir: Where to save outputs
            include_social_media: Whether to include social media chains
            
        Returns:
            Dict mapping variant_name -> (output_path, metadata)
        """
        results = {}
        save_dir = output_dir or self.output_dir
        
        # Generate codec ladder for each codec
        for codec in self.CODECS:
            ladder = self.generate_codec_ladder(input_path, codec, save_dir)
            for bitrate, (path, meta) in ladder.items():
                variant_name = f"{codec}_{bitrate}k"
                results[variant_name] = (path, meta)
        
        # Add social media chains
        if include_social_media:
            for platform in self.SOCIAL_MEDIA_CHAINS:
                output_path, metadata = self.apply_social_media_chain(
                    input_path, platform,
                    output_path=None if not save_dir else
                              str(save_dir / f"{Path(input_path).stem}_{platform}.{self.SOCIAL_MEDIA_CHAINS[platform]['extension']}")
                )
                results[platform] = (output_path, metadata)
        
        return results


if __name__ == '__main__':
    # Example usage
    print("Codec Simulator - Testing ffmpeg installation...")
    
    simulator = CodecSimulator()
    print("✓ ffmpeg found and working")
    
    print("\nAvailable codecs:")
    for codec, info in CodecSimulator.CODECS.items():
        print(f"  {codec}: {info['bitrates']} kbps")
    
    print("\nAvailable social media chains:")
    for platform, info in CodecSimulator.SOCIAL_MEDIA_CHAINS.items():
        print(f"  {platform}: {info['description']}")
