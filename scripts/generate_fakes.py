"""Generate fake samples from real audio."""
import subprocess
from pathlib import Path
from tqdm import tqdm

real_dir = Path("data/raw/auto_download/train_data/real")
fake_dir = Path("data/raw/auto_download/train_data/fake")
fake_dir.mkdir(exist_ok=True)

real_files = sorted(list(real_dir.glob("*.wav")))[:500]

print(f"Generating {len(real_files)} fake samples...")

for i, real_file in enumerate(tqdm(real_files)):
    output = fake_dir / f"fake_{i:04d}.wav"
    
    cmd = [
        'ffmpeg', '-nostdin', '-y', '-i', str(real_file),
        '-af', 'aresample=8000,aresample=16000',
        '-ar', '16000', '-ac', '1',
        '-c:a', 'pcm_s16le',  # Use WAV format instead
        str(output)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     check=True, timeout=10)
    except:
        continue

fake_count = len(list(fake_dir.glob("*.wav")))
print(f"\nGenerated {fake_count} fake samples")
