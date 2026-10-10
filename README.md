# talking-prop

Live voice conversation for animatronic props. Talk to a skull, a bird, a puppet or anything else with a servo jaw, and it answers out loud with its jaw moving in time to its own voice.

A Raspberry Pi handles the microphone, speaker and servo. All the heavy work (speech recognition, the language model and the synthesized voice) runs on a separate speech-to-speech server on your network, so even a Pi 3B+ is plenty.

```
 mic ──► Pi ──websocket──► speech-to-speech server ──► LLM
                                │        (STT + TTS)
 speaker ◄── Pi ◄── reply audio ┘
   └─► jaw servo follows the loudness of the reply as it plays
```

## Contents

- [Features](#features)
- [What you need](#what-you-need)
- [Wiring](#wiring)
- [Server setup](#server-setup)
- [Install on the Pi](#install-on-the-pi)
- [Run it](#run-it)
- [Moving eye](#moving-eye)
- [Personality](#personality)
- [Web search](#web-search)
- [Voices](#voices)
- [Microphones](#microphones)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Tests](#tests)
- [Credits](#credits)

## Features

- **Hands-free conversation.** No wake word: the server detects when someone starts and stops talking.
- **Jaw sync.** The servo follows the reply audio as it plays, using the same loudness-to-jaw method as [ChatterPi](https://github.com/ViennaMike/ChatterPi). An existing ChatterPi `config.ini` works unchanged.
- **Personality from a text file.** `persona.txt` sets who the prop is and how it talks.
- **Optional web search** through the [Brave Search API](https://brave.com/search/api/), with SafeSearch set to strict.
- **Self-muting.** The mic is muted while the prop speaks so it doesn't answer itself.
- **Optional moving eye** on two servos (side to side and up and down) that glances around while the prop talks.
- **Optional LED eyes** that light while the prop talks.
- **Reconnects on its own** if the server or network drops.

## What you need

**On the prop**

- A Raspberry Pi. Tested on a Pi 3B+ with Raspberry Pi OS Lite (64-bit).
- A hobby servo on the jaw, with its own 5 V supply.
- A microphone and speaker. A USB audio adapter with a mic jack is the simplest; the Pi's own 3.5 mm jack is output only. See [Microphones](#microphones).
- Optional: LEDs for eyes.

**On your network**

- A machine with an NVIDIA GPU running a speech-to-speech server that speaks the OpenAI-style realtime websocket protocol at `/v1/realtime` with 16 kHz PCM audio. This project was built against [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) in realtime mode; see [Server setup](#server-setup).
- An OpenAI-compatible LLM endpoint for the server to use, such as llama.cpp's `llama-server` or Ollama.

## Wiring

| Part | Pi pin (BCM) | Notes |
|---|---|---|
| Jaw servo signal | GPIO 18 | Set by `jaw_pin` in `config.ini`. |
| Jaw servo power | External 5 V | Don't power the servo from the Pi's 5 V pin; it browns out the Pi. |
| Servo ground | Any GND | Must be shared between the servo supply and the Pi. |
| LED eyes (optional) | GPIO 25 | Through a resistor. Set `eyes = ON` and `eyes_pin` in `config.ini`. |
| Eye pan servo (optional) | GPIO 12 | Side to side. Set in `[EYE_SERVOS]`; see [Moving eye](#moving-eye). |
| Eye tilt servo (optional) | GPIO 13 | Up and down. |

## Server setup

The Pi talks to the speech-to-speech server; the server does speech recognition, calls your LLM and speaks the reply in a cloned voice. An example `docker-compose.yml` for Hugging Face speech-to-speech, using Parakeet for speech recognition, Qwen3-TTS for the voice, and an LLM served by `llama-server` elsewhere on the network:

```yaml
services:
  pipeline:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=unused
    command:
      - speech-to-speech
      - --mode
      - realtime
      - --recv_host
      - 0.0.0.0
      - --send_host
      - 0.0.0.0
      - --stt
      - parakeet-tdt
      - --tts
      - qwen3
      - --llm_backend
      - chat-completions
      - --model_name
      - your-model-name
      - --responses_api_base_url
      - http://192.168.1.50:9931/v1
      - --qwen3_tts_model_name
      - Qwen/Qwen3-TTS-12Hz-1.7B-Base
      - --qwen3_tts_language
      - en
      - --qwen3_tts_ref_audio
      - /root/.cache/voice.wav
      - --qwen3_tts_ref_text
      - "Exactly what is said in voice.wav, on one line."
    ports:
      - 8765:8765/tcp
    volumes:
      - ./cache/:/root/.cache/
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

Notes:

- **Clone the speech-to-speech repo first** and put this file in it; `build` uses its Dockerfile.
- **The voice clip goes in `./cache/`**, which the container sees as `/root/.cache/`. See [Voices](#voices) for making one.
- **`--qwen3_tts_ref_text` must be a single string on one line.** YAML won't join quoted pieces spread across lines.
- **`restart: unless-stopped`** brings the server back after a reboot.
- **One client at a time** by default. A second prop or robot connecting at the same time is refused unless you add `--num_pipelines 2`, which loads a second set of models into GPU memory.
- **Thinking models:** turn thinking off on the LLM, or the reasoning text can end up spoken.

Check the config before starting, then watch it come up:

```bash
docker compose config >/dev/null && echo OK
docker compose up -d
docker compose logs -f
```

## Install on the Pi

Flash **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager, setting the hostname, user, Wi-Fi and SSH in its settings. Then:

```bash
sudo apt update
sudo apt install -y git python3-pyaudio python3-numpy python3-gpiozero python3-pigpio python3-websocket
git clone https://github.com/stoney66/talking-prop.git
cd talking-prop
cp config.example.ini config.ini
cp persona.example.txt persona.txt
```

If you already use ChatterPi, copy its `config.ini` instead of the example; your servo calibration carries over.

### pigpio (smooth servo movement)

Without the pigpio daemon the servo pulses are timed in software and the jaw may twitch. The script still runs without it and prints a notice at startup.

If `sudo apt install pigpio` works on your OS release, use that and run `sudo systemctl enable --now pigpiod`. Recent releases don't package it, so build it from source:

```bash
sudo apt install -y build-essential
cd ~ && git clone https://github.com/joan2937/pigpio.git
cd pigpio && make
sudo make install   # an error about the Python module ("No module named 'distutils'") at the end is fine
sudo ldconfig
cd ~/talking-prop
sudo cp pigpiod.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pigpiod
systemctl is-active pigpiod   # should print: active
```

### Audio

Make your USB audio device the system default, by **name** rather than card number, because card numbers can change between boots:

```bash
cat /proc/asound/cards
```

The name is the word in square brackets, e.g. `[Device         ]` gives `Device`. Then:

```bash
cat > ~/.asoundrc <<'EOF'
pcm.!default {
    type asym
    playback.pcm { type plug; slave.pcm "hw:Device,0" }
    capture.pcm  { type plug; slave.pcm "hw:Device,0" }
}
ctl.!default { type hw; card Device }
EOF
```

If the mic and speaker are separate devices, use each one's name: the mic's in `capture.pcm` and the speaker's in `playback.pcm`.

With that in place, `talking_prop.py` needs no device options. Check what it sees with `python3 talking_prop.py --list-devices`; the `default` entry should show both `in=` and `out=` above 0. The ALSA and JACK warnings printed above the list are harmless.

### Mic level

Watch the live level while speaking at the distance people will use:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -V mono /dev/null
```

- Normal voice: peaks around 30 to 60%.
- Shouting: under 90%. At 99 to 100% it's clipping; turn the gain down.
- Silence: near 0 to 2%.

Adjust with `alsamixer -c Device` (F4 for capture controls; press M on Auto Gain Control to turn it off if listed). With an XLR interface, set its hardware gain knob first. Listen back with:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 test.wav && aplay test.wav
```

Save the mixer level across reboots with `sudo alsactl store`.

## Run it

Check the jaw. It should step open and closed three times:

```bash
python3 talking_prop.py --jaw-test
```

Then talk to it:

```bash
python3 talking_prop.py --url ws://<server-ip>:8765/v1/realtime
```

You'll see `Connected to ...`, then `You:` and `Prop:` lines as the conversation goes.

### Options

| Option | What it does |
|---|---|
| `--url` | Websocket URL of the speech-to-speech server (or set `TALKING_PROP_URL`). Required. |
| `--voice` | Voice to request from the server. What it means is up to the server; see [Voices](#voices). |
| `--persona` | Personality file (default `persona.txt`). |
| `--config` | Servo config file (default `config.ini`). |
| `--barge-in` | Keep the mic live while the prop talks, so people can interrupt. Only with a mic that won't pick up the prop's own speaker. |
| `--mic-gain` | Multiply the mic signal in software, e.g. `3` for a quiet headset mic. Raise the mixer level first; use this only if that's already at maximum. |
| `--tail` | Seconds the mic stays muted after the prop stops talking (default 0.4). Raise it, e.g. to 0.8, for loud speakers or echoey spaces. |
| `--mic-device`, `--out-device` | Audio device by number or part of its name. Usually not needed; see [Audio](#audio). |
| `--mic-rate`, `--out-rate` | Device sample rate: 16000, 32000 or 48000 (default 16000). |
| `--language` | Speech recognition language (default `en`). |
| `--token` | Bearer token, if your server requires one (or set `TALKING_PROP_TOKEN`). |
| `--brave-key-file` | Where to find the Brave key (default `brave_key.txt`). |
| `--no-search-fallback` | Don't search on the model's behalf when it says it will look something up but doesn't call the tool. |
| `--kid-safe` | Keep answers family-friendly: adds a rule to the instructions and tells the model to skip crime, violence and other upsetting items in search results. Recommended around children. |
| `--clock-refresh` | Seconds between updates of the time given to the model (default 60). |
| `--no-search` | Turn web search off even if a key is present. |
| `--no-servo` | Run without touching GPIO, e.g. on a desktop. |
| `--list-devices` | Print audio devices and exit. |
| `--eye-test [pan\|tilt]` | Move the eye servos through their range and exit. Add `pan` or `tilt` to move just that one. |
| `--jaw-test` | Step the jaw through its positions and exit. |
| `--debug` | Print every event the server sends, plus a mic level meter once a second. |

### Start at boot

Edit `User`, `WorkingDirectory` and the `--url` in `talking-prop.service` (add any other options to the end of `ExecStart`), then:

```bash
sudo cp talking-prop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now talking-prop
```

Day to day:

| Task | Command |
|---|---|
| Watch the conversation | `journalctl -u talking-prop -f` |
| Restart after editing `persona.txt` or `config.ini` | `sudo systemctl restart talking-prop` |
| Stop it (before running the script by hand) | `sudo systemctl stop talking-prop` |

The service and a hand-run copy can't share the mic and servo, so stop one before starting the other.

## Moving eye

An eye on two servos, one for side to side (pan) and one for up and down (tilt), glances to random spots while the prop talks and returns to centre when it stops. Each servo needs its own GPIO pin: a Y-cable would send both the same signal, so the eye could only move diagonally.

Power both servos from the same external 5 V supply as the jaw, with grounds shared with the Pi. Then add to `config.ini` (the full set of keys, with comments, is in `config.example.ini`):

```ini
[EYE_SERVOS]
enabled = ON
pan_pin = 12
tilt_pin = 13
pan_range = 0.8
tilt_range = 0.5
```

Check the movement and limits:

```bash
python3 talking_prop.py --eye-test
```

It looks left, right, up, down, then around the corners, and centres. To test one servo on its own, use `--eye-test pan` or `--eye-test tilt`; the other servo gets no signal. If a direction is reversed, set `pan_reverse = ON` or `tilt_reverse = ON`. If the eye hits the edge of the socket, lower `pan_range` or `tilt_range`, or narrow the pulse widths. `min_hold` and `max_hold` set how long it holds each glance.

## Personality

`persona.txt` is sent to the model as its instructions each time the Pi connects. The example, `persona.example.txt`, is a pirate skull. Rules that work well for a prop:

- **Keep replies to one or two short sentences.** Everything is spoken aloud, and long replies take longer to synthesize.
- **Ban lists, emoji, markdown and stage directions.** They get read out literally.
- **Put the most important rule first.** Smaller models follow early rules more reliably; "never ask the listener a question" works better at the top.
- **Write the dialect into the rules** (e.g. "ye" for "you") to reinforce an accent in the cloned voice.

The current date and time are added automatically and refreshed every minute while the prop is idle (`--clock-refresh`), so "what time is it?" works even with models that rarely call tools. A `get_current_time` tool is offered as well.

## Web search

Put a [Brave Search API](https://brave.com/search/api/) key in `brave_key.txt` beside the script (or set `BRAVE_API_KEY`):

```bash
echo 'YOUR_KEY' > brave_key.txt && chmod 600 brave_key.txt
```

The prop then gets a `web_search` tool and uses it for current information: weather, news, scores, opening hours. Each search shows in the log as `Search: ...`. Results reach the model as title, snippet and site name only, so it can't read out web addresses. SafeSearch is always strict.

Smaller models sometimes say they'll look something up ("I'll check the latest from Oregon...") without actually calling the tool. When a reply sounds like that and no search happened, talking-prop runs the search itself with the visitor's question and has the model answer from the results, so the prop follows its promise with a real answer. Turn this off with `--no-search-fallback`. A persona rule such as "For anything about today, the weather or the news, use web_search first" also helps.

`brave_key.txt` is in `.gitignore`; keep it out of any repo.

## Voices

The voice comes from the speech-to-speech server, not from the Pi. With Qwen3-TTS, the server clones a voice from a short reference clip and its exact transcript (`--qwen3_tts_ref_audio` and `--qwen3_tts_ref_text`).

### Getting a reference clip

- **Record one.** 10 to 20 seconds of one clean voice, no music or background noise, read from a written line so the transcript is exact. A real recording carries an accent best.
  ```bash
  arecord -D default -f S16_LE -r 24000 -c 1 -d 15 voice.wav
  ```
- **Use a public-domain recording.** For example, a LibriVox reading; cut a clean 15 seconds with `ffmpeg -i chapter.mp3 -ss 00:02:10 -t 15 -ac 1 -ar 24000 voice.wav` and type out exactly what's said. Avoid film clips; music under dialogue clones badly.
- **Design one from a description** with [`tools/make_voice.py`](tools/make_voice.py), which uses the `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` model to write several candidate clips. Run it on the GPU machine, not the Pi, with the speech-to-speech server stopped to free GPU memory.

Tips for designed voices:

- Describe the sound (age, pitch, roughness, pace, emotion), not the character. "Pirate" alone does little.
- Change one thing per round and rerun; the same description gives a slightly different voice each time.
- Accents are the weak point. Spelling the accent into the text (`Oi've`, `ev'ry`, `'ere`) helps a lot; if it still isn't enough, record a real voice instead.

### Installing it

1. Copy the clip into the server's `cache` folder, e.g. `cache/voice.wav`, which the container sees as `/root/.cache/voice.wav`.
2. Set `--qwen3_tts_ref_audio` to that path and `--qwen3_tts_ref_text` to the clip's exact transcript, on one line.
3. `docker compose up -d` (not `docker restart`, which ignores compose changes).

To switch between voices, keep the other pair commented out in the compose file and swap them. The server also accepts a per-connection voice (`--voice` on the Pi) given as the path of a clip on the server, but it keeps one transcript for every voice, so swapping the server's pair is more reliable.

## Microphones

- **Placement matters more than price.** Put the mic where people stand, not inside the prop, and point it away from the speaker.
- **Handheld dynamic mic on a stand** (e.g. a Shure PG58, through an XLR-to-USB interface): the most reliable choice for a crowd. Only the person at the mic is heard, and it rejects the prop's own speaker, which may let you use `--barge-in`. A "speak to the captain" sign does the rest. Aim the back of the mic at the speaker.
- **Desk testing:** any plug-in-power PC mic with a 3-pole (TRS) 3.5 mm plug works in a USB adapter's mic jack. Phone headset mics use 4-pole (TRRS) plugs and need a splitter.
- **Hidden far-field mic:** a USB mic array or conference speakerphone picks up people a few feet away, but will also hear everyone else nearby.
- **Outdoors:** use a foam windscreen and keep the electronics covered.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Can't connect to pigpio at localhost(8888)` then `pigpio unavailable` | The pigpio daemon isn't running. Harmless (software timing is used); see [pigpio](#pigpio-smooth-servo-movement) for smooth movement. |
| Lots of `ALSA lib ...` and `jack server is not running` lines | Harmless noise from the audio libraries probing devices. |
| `invalid int value: 'default'` | An old copy of the script; this version accepts names. Or leave the device options off. |
| `Segmentation fault` or `capture slave is not defined` after a reboot | The USB card number changed. Use the name-based `~/.asoundrc` from [Audio](#audio) and drop device numbers. |
| `--list-devices` shows `in=0` everywhere | No microphone detected. The Pi's 3.5 mm jack has no mic input; use USB. Check `arecord -l` and `lsusb`. |
| `No server URL` | Pass `--url` or set `TALKING_PROP_URL`. |
| `Disconnected; retrying in 3 seconds` repeating | Server down, wrong URL, or another client already connected (one at a time by default). |
| It answers itself | The mic hears the speaker. Raise `--tail`, turn the speaker down, move the mic, and don't use `--barge-in` with an omnidirectional mic. |
| It doesn't hear people, or hears every noise | Check the [mic level](#mic-level). |
| Replies are long or ignore persona rules | Shorten the persona and put the key rule first. |
| It never searches | No key found (the connect line says `web search off`), or the model prefers answering from memory; add a search rule to the persona. |
| It says "I'll check" and then answers a moment later | That's the search fallback covering for a model that didn't call the tool. Normal. |
| The voice reads out `<think>` or reasoning | Turn thinking off on the LLM. |
| Voice design: `CUDA out of memory` | Stop the speech-to-speech server while generating, or use another GPU (`device_map="cuda:1"`). |
| Server won't start after editing the transcript | The YAML transcript must be one quoted string on one line. Check with `docker compose config`. |

## How it works

- **Session.** On connect, the Pi sends a `session.update` with the persona, the date, the voice, server-side voice activity detection and, if a Brave key is present, the `web_search` tool. Audio is 16 kHz mono 16-bit PCM in both directions.
- **Listening.** Mic audio streams continuously as `input_audio_buffer.append`. While the prop is talking (plus `--tail` seconds), silence is sent instead, unless `--barge-in` is on.
- **Speaking.** Reply audio arrives as `response.output_audio.delta` chunks and is queued for playback. Every 20 ms of playback, the average loudness of that chunk picks one of four jaw positions (or two with `style = 0`), so the jaw tracks what is actually coming out of the speaker.
- **Searching.** When the model calls `web_search`, the Pi queries Brave, waits for the current response to finish, returns the results as a `function_call_output`, and asks the model to continue.
- **Eye.** While the prop talks, the eye eases towards a random spot every half-second to second and a half; afterwards it centres and its servos are released too.
- **Resting.** After speaking, the jaw closes and the servo pulses stop so it doesn't buzz.

## Tests

No hardware needed. They cover the jaw mapping, resampling, device lookup, Brave result parsing, and full conversations against a fake realtime server:

```bash
pip install pytest websockets numpy websocket-client
python3 -m pytest tests
```

## Credits

- [ChatterPi](https://github.com/ViennaMike/ChatterPi) by Mike McGurrin, for the loudness-to-jaw method and the `config.ini` format this project reads.
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) by Pollen Robotics, whose realtime session setup this client follows.
- [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech), the server it was built against.
- [Qwen3-TTS](https://huggingface.co/Qwen) for the cloned and designed voices.

## License

MIT. See [LICENSE](LICENSE).
