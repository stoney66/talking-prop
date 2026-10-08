#!/usr/bin/env python3
"""Generate candidate reference clips for a cloned TTS voice with Qwen3-TTS VoiceDesign.

Run on the machine with the GPU (not the Pi):

    python3 -m venv ~/voicedesign && source ~/voicedesign/bin/activate
    pip install qwen-tts soundfile
    python3 make_voice.py

Listen to the voice_*.wav files, adjust VOICES and run again. Use the clip you like as the
server's --qwen3_tts_ref_audio and TEXT, exactly as written, as --qwen3_tts_ref_text.
Free GPU memory first (stop the speech-to-speech server) if the model doesn't fit.
"""
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# What the voice says in the clip. Spell an accent into it; the clone copies how it's spoken.
TEXT = ("Arrr, gather 'round an' listen well, me hearties. Oi've sailed ev'ry sea "
        "on this 'ere earth, buried more gold than a king could spend, an' Oi "
        "b'ain't done yet, arrr!")

# Describe the sound (age, pitch, roughness, pace, emotion), not the character.
BASE = ("Male, around fifty. Low, booming, theatrical stage voice. Rolls his Rs "
        "hard, stretches vowels, swings between a whisper and a roar. ")
VOICES = {
    "1": BASE + "Thick rural English accent: hard growled R after every vowel, "
                "dropped Hs, broad drawn-out vowels.",
    "2": BASE + "Rougher and hoarser, with a growl under every word and a heavy rolling R.",
    "3": BASE + "Slower and more menacing, lingering on each word, ending phrases with a low growl.",
    "4": BASE,  # the same description again; every run gives a slightly different voice
}

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                      device_map="cuda:0", dtype=torch.bfloat16)
for name, description in VOICES.items():
    wavs, rate = model.generate_voice_design(text=TEXT, language="English", instruct=description)
    sf.write("voice_%s.wav" % name, wavs[0], rate)
    print("wrote voice_%s.wav" % name)
