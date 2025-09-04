import os
import torch
from pathlib import Path
from TTS.api import TTS

# === Paths (update SPEAKER_REF only) ===
ROOT = Path("/Users/dovudkhon/Desktop/Coding/RussianTTS").resolve()

# Fine-tuned checkpoint (you gave this path)
FT_MODEL = ROOT / "run/training/GPT_XTTS_v2.0_LJSpeech_FT-September-04-2025_12+12AM-0000000/model.pth"
FT_CONFIG = FT_MODEL.parent / "config.json"   # trainer saved config next to the checkpoint

# Base model (original files folder; needs model.pth + mel_stats.pth + dvae.pth + vocab.json in same dir)
BASE_MODEL = ROOT / "run/training/XTTS_v2.0_original_model_files/model.pth"
BASE_CONFIG = BASE_MODEL.parent / "config.json"  # if you don't have this, the trainer's original config works too

# A short reference wav from your dataset (ABSOLUTE PATH!)
SPEAKER_REF = Path("/Users/dovudkhon/Desktop/Coding/RussianTTS/ruls_data/train/audio/9014/11018/eugeneonegin_01_pushkin_0067.wav").resolve()

# Output folder
OUT = ROOT / "inference_out"
OUT.mkdir(parents=True, exist_ok=True)

# Sanity checks
for p in [FT_MODEL, FT_CONFIG, BASE_MODEL, BASE_CONFIG, SPEAKER_REF]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing required path: {p}")

# Pick device: use MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Load fine-tuned model
tts_ft = TTS(model_path=str(FT_MODEL.parent), config_path=str(FT_CONFIG), progress_bar=False, gpu=False)

# Load base model
tts_base = TTS(model_path=str(BASE_MODEL.parent), config_path=str(BASE_CONFIG), progress_bar=False, gpu=False)

# Russian test lines (shorter than 182 chars)
tests = [
    "Москва раскинулась на обоих берегах реки и живёт в ритме большого мегаполиса.",
    "Современные технологии постепенно меняют привычный уклад нашей жизни.",
    "Тихий ветер колышет листву деревьев, и в воздухе чувствуется запах осени.",
    "Каждый человек ищет вдохновение в музыке, книгах или простых разговорах.",
    "История хранит множество примеров мужества, мудрости и стремления к свободе.",
]

def synth_pair(idx: int, text: str):
    ft_path = OUT / f"ft2_{idx:02d}.wav"
    base_path = OUT / f"base2_{idx:02d}.wav"

    # Fine-tuned
    tts_ft.tts_to_file(
        text=text,
        speaker_wav=str(SPEAKER_REF),
        language="ru",
        file_path=str(ft_path)
    )

    # Base
    tts_base.tts_to_file(
        text=text,
        speaker_wav=str(SPEAKER_REF),
        language="ru",
        file_path=str(base_path)
    )

    print(f"[OK] {text[:52]}... -> {ft_path.name} / {base_path.name}")

for i, line in enumerate(tests, 1):
    synth_pair(i, line)

print(f"\nDone. Files written to: {OUT}")