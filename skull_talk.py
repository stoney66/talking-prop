#!/usr/bin/env python3
"""skull_talk.py - live speech-to-speech conversation for a talking skull.

Mic -> realtime s2s server (same /v1/realtime protocol the Reachy Mini
conversation app uses) -> speaker, with the jaw servo driven by the volume
of the reply audio the same way ChatterPi does it.

Files read from the script's own directory:
  config.ini   ChatterPi's config.ini (servo calibration, levels, pins). Optional.
  persona.txt  The personality / instructions sent to the model. Optional.

Run "python3 skull_talk.py --help" for options.
"""
import argparse
import base64
import configparser
import json
import os
import queue
import threading
import time

import numpy as np

SERVER_RATE = 16000  # the s2s server speaks 16 kHz mono 16-bit PCM both ways
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PERSONA = (
    "You are a talking skeleton. Keep every reply to one or two short spoken "
    "sentences. Never use lists, emoji or stage directions."
)


# --------------------------------------------------------------------------
# Settings carried over from ChatterPi's config.ini
# --------------------------------------------------------------------------
def load_chatterpi_config(path):
    """Return jaw settings, using ChatterPi's defaults for anything missing."""
    cfg = configparser.ConfigParser()
    cfg.read(path)

    def get(section, key, default):
        try:
            return cfg[section][key]
        except KeyError:
            return default

    return {
        "servo_min": int(get("SERVO", "servo_min", 1050)),
        "servo_max": int(get("SERVO", "servo_max", 1250)),
        "min_angle": int(get("SERVO", "min_angle", 0)),
        "max_angle": int(get("SERVO", "max_angle", 90)),
        "style": int(get("CONTROLLER", "style", 1)),
        "threshold": int(get("CONTROLLER", "threshold", 2000)),
        "level1": int(get("CONTROLLER", "level1", 1500)),
        "level2": int(get("CONTROLLER", "level2", 2500)),
        "level3": int(get("CONTROLLER", "level3", 3500)),
        "eyes": str(get("PROP", "eyes", "OFF")).upper() == "ON",
        "jaw_pin": int(get("PINS", "jaw_pin", 18)),
        "eyes_pin": int(get("PINS", "eyes_pin", 25)),
    }


# --------------------------------------------------------------------------
# Jaw
# --------------------------------------------------------------------------
class Jaw:
    """Maps the loudness of a chunk of audio to a servo angle (ChatterPi's method)."""

    def __init__(self, settings, enabled=True):
        self.s = settings
        # Same convention as ChatterPi: the larger angle is the resting position.
        self.closed = max(settings["min_angle"], settings["max_angle"])
        self.open = min(settings["min_angle"], settings["max_angle"])
        self.servo = None
        self.eyes = None
        self.last_angle = None
        if not enabled:
            return
        from gpiozero import AngularServo, DigitalOutputDevice

        factory = None
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory

            factory = PiGPIOFactory()  # hardware-timed pulses: no servo jitter
        except Exception as exc:  # pigpiod not running or not installed
            print("pigpio unavailable (%s); using gpiozero's default pins. "
                  "Expect some servo jitter. Fix: sudo systemctl enable --now pigpiod" % exc)
        self.servo = AngularServo(
            settings["jaw_pin"],
            min_angle=settings["min_angle"],
            max_angle=settings["max_angle"],
            initial_angle=None,
            min_pulse_width=settings["servo_min"] / 1e6,
            max_pulse_width=settings["servo_max"] / 1e6,
            pin_factory=factory,
        )
        if settings["eyes"]:
            self.eyes = DigitalOutputDevice(settings["eyes_pin"], pin_factory=factory)

    def target(self, samples):
        """Return the jaw angle for one chunk of int16 samples."""
        if len(samples) == 0:
            return self.closed
        volume = int(np.abs(samples.astype(np.int32)).mean())
        s = self.s
        if s["style"] == 0:  # single threshold: open or shut
            return self.open if volume > s["threshold"] else self.closed
        step = (self.open - self.closed) / 3.0
        if volume > s["level3"]:
            return self.open
        if volume > s["level2"]:
            return self.closed + 2 * step
        if volume > s["level1"]:
            return self.closed + step
        return self.closed

    def move(self, angle):
        if angle == self.last_angle:
            return
        self.last_angle = angle
        if self.servo is not None:
            self.servo.angle = angle

    def speaking(self, on):
        if self.eyes is not None:
            self.eyes.on() if on else self.eyes.off()

    def rest(self):
        """Close the jaw, then stop sending pulses so the servo does not buzz."""
        self.move(self.closed)
        if self.servo is not None:
            time.sleep(0.15)
            self.servo.angle = None
        self.last_angle = None


# --------------------------------------------------------------------------
# Sample-rate helpers (whole-number ratios only; cheap enough for a Pi 3)
# --------------------------------------------------------------------------
def rate_factor(device_rate):
    if device_rate % SERVER_RATE:
        raise SystemExit("Rate %d is not a multiple of %d. Use 16000, 32000 or 48000."
                         % (device_rate, SERVER_RATE))
    return device_rate // SERVER_RATE


def downsample(samples, factor):
    if factor == 1:
        return samples
    usable = len(samples) - len(samples) % factor
    return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.int16)


def upsample(samples, factor):
    return samples if factor == 1 else np.repeat(samples, factor)


# --------------------------------------------------------------------------
# The conversation
# --------------------------------------------------------------------------
class SkullTalk:
    def __init__(self, args, jaw):
        self.args = args
        self.jaw = jaw
        self.mic_factor = rate_factor(args.mic_rate)
        self.out_factor = rate_factor(args.out_rate)
        self.mic_queue = queue.Queue(maxsize=50)
        self.play_buffer = bytearray()  # 16 kHz PCM waiting to be played
        self.play_lock = threading.Lock()
        self.last_audio_time = 0.0
        self.is_speaking = False
        self.ws = None
        self.connected = threading.Event()
        self.stop = threading.Event()

    # ---- audio device callbacks (run on PyAudio's thread) ----
    def on_mic(self, in_data, frame_count, time_info, status):
        try:
            self.mic_queue.put_nowait(in_data)
        except queue.Full:
            pass  # not connected or network stalled: drop audio rather than lag
        return (None, 0)  # 0 == pyaudio.paContinue

    def on_speaker(self, in_data, frame_count, time_info, status):
        want = (frame_count // self.out_factor) * 2  # bytes of 16 kHz audio
        with self.play_lock:
            chunk = bytes(self.play_buffer[:want])
            del self.play_buffer[:want]
        samples = np.frombuffer(chunk, dtype=np.int16)
        if len(samples):
            self.last_audio_time = time.monotonic()
            self.jaw.move(self.jaw.target(samples))
        if len(chunk) < want:
            samples = np.concatenate([samples, np.zeros((want - len(chunk)) // 2, dtype=np.int16)])
        return (upsample(samples, self.out_factor).tobytes(), 0)

    def skull_is_talking(self):
        """True while reply audio is playing, plus a short tail for room echo."""
        with self.play_lock:
            pending = len(self.play_buffer) > 0
        return pending or (time.monotonic() - self.last_audio_time) < self.args.tail

    # ---- websocket ----
    def session_update(self):
        output = {"format": {"type": "audio/pcm", "rate": None}}
        if self.args.voice:
            output["voice"] = self.args.voice
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self.args.persona_text,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": None},
                        "transcription": {"model": "gpt-4o-transcribe", "language": self.args.language},
                        "turn_detection": {"type": "server_vad", "interrupt_response": True},
                    },
                    "output": output,
                },
                "tools": [],
                "tool_choice": "auto",
            },
        }

    def on_open(self, ws):
        ws.send(json.dumps(self.session_update()))
        self.connected.set()
        print("Connected to %s (voice=%s)" % (self.args.url, self.args.voice or "server default"))

    def on_message(self, ws, message):
        try:
            event = json.loads(message)
        except ValueError:
            return
        kind = event.get("type", "")
        if kind in ("response.output_audio.delta", "response.audio.delta"):
            pcm = base64.b64decode(event.get("delta", ""))
            with self.play_lock:
                self.play_buffer.extend(pcm)
        elif kind == "input_audio_buffer.speech_started":
            if self.args.barge_in:  # you spoke over the skull: stop it
                with self.play_lock:
                    del self.play_buffer[:]
        elif kind == "conversation.item.input_audio_transcription.completed":
            print("You:   %s" % (event.get("transcript") or "").strip())
        elif kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
            print("Skull: %s" % (event.get("transcript") or "").strip())
        elif kind == "error":
            print("Server error: %s" % json.dumps(event.get("error", event)))
        elif self.args.debug:
            print("event: %s" % kind)

    def on_close(self, ws, *unused):
        self.connected.clear()

    def on_error(self, ws, error):
        print("Connection problem: %s" % error)

    def run_socket(self):
        import websocket  # the "websocket-client" package

        while not self.stop.is_set():
            self.ws = websocket.WebSocketApp(
                self.args.url,
                header=["Authorization: Bearer %s" % self.args.token],
                on_open=self.on_open,
                on_message=self.on_message,
                on_close=self.on_close,
                on_error=self.on_error,
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            self.connected.clear()
            if not self.stop.is_set():
                print("Disconnected; retrying in 3 seconds")
                self.stop.wait(3)

    def send_mic(self):
        """Forward mic audio; send silence while the skull talks unless barge-in is on."""
        while not self.stop.is_set():
            try:
                raw = self.mic_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.connected.is_set():
                continue
            samples = downsample(np.frombuffer(raw, dtype=np.int16), self.mic_factor)
            if not self.args.barge_in and self.skull_is_talking():
                samples = np.zeros_like(samples)
            message = {"type": "input_audio_buffer.append",
                       "audio": base64.b64encode(samples.tobytes()).decode("ascii")}
            try:
                self.ws.send(json.dumps(message))
            except Exception:
                self.connected.clear()

    def watch_speaking(self):
        """Light the eyes while talking and relax the servo afterwards."""
        while not self.stop.is_set():
            talking = self.skull_is_talking()
            if talking and not self.is_speaking:
                self.jaw.speaking(True)
            elif not talking and self.is_speaking:
                self.jaw.speaking(False)
                self.jaw.rest()
            self.is_speaking = talking
            time.sleep(0.05)

    def run(self):
        import pyaudio

        pa = pyaudio.PyAudio()
        mic_index = find_device(pa, self.args.mic_device, need_input=True)
        out_index = find_device(pa, self.args.out_device, need_input=False)
        mic = pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.mic_rate, input=True,
                      input_device_index=mic_index,
                      frames_per_buffer=int(self.args.mic_rate * 0.04), stream_callback=self.on_mic)
        speaker = pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.out_rate, output=True,
                          output_device_index=out_index,
                          frames_per_buffer=int(self.args.out_rate * 0.02), stream_callback=self.on_speaker)
        threads = [threading.Thread(target=fn, daemon=True)
                   for fn in (self.run_socket, self.send_mic, self.watch_speaking)]
        for thread in threads:
            thread.start()
        print("Listening. Ctrl-C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            if self.ws is not None:
                self.ws.close()
            mic.stop_stream()
            speaker.stop_stream()
            mic.close()
            speaker.close()
            pa.terminate()
            self.jaw.rest()


def find_device(pa, wanted, need_input):
    """Turn a device number or a piece of its name into a PyAudio device index."""
    if wanted is None:
        return None
    key = "maxInputChannels" if need_input else "maxOutputChannels"
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    if str(wanted).isdigit():
        index = int(wanted)
        if index < len(devices) and devices[index][key] > 0:
            return index
        raise SystemExit("Device %s is not a usable %s. Run --list-devices; numbers can change after a reboot."
                         % (wanted, "microphone" if need_input else "speaker"))
    usable = [d for d in devices if d[key] > 0]
    exact = [d for d in usable if d["name"].lower() == str(wanted).lower()]
    partial = [d for d in usable if str(wanted).lower() in d["name"].lower()]
    for match in exact + partial:
        return int(match["index"])
    raise SystemExit("No %s matching %r. Run --list-devices." % ("microphone" if need_input else "speaker", wanted))


def list_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        print("%2d  in=%d out=%d  %s" % (index, info["maxInputChannels"],
                                          info["maxOutputChannels"], info["name"]))
    pa.terminate()


def jaw_test(jaw):
    """Step the jaw through its four positions so the calibration can be checked."""
    step = (jaw.open - jaw.closed) / 3.0
    for _ in range(3):
        for position in (0, 1, 2, 3, 2, 1):
            jaw.move(jaw.closed + position * step)
            time.sleep(0.25)
    jaw.rest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live conversation for a talking skull.")
    p.add_argument("--url", default=os.getenv("SKULL_WS_URL", "ws://192.168.86.10:8765/v1/realtime"),
                   help="realtime websocket URL of the s2s server")
    p.add_argument("--token", default=os.getenv("SKULL_TOKEN", "DUMMY"), help="bearer token, if the server wants one")
    p.add_argument("--voice", default=os.getenv("SKULL_VOICE", ""), help="voice name known to the s2s server")
    p.add_argument("--language", default="en", help="transcription language")
    p.add_argument("--persona", default=os.path.join(HERE, "persona.txt"), help="file with the personality text")
    p.add_argument("--config", default=os.path.join(HERE, "config.ini"), help="ChatterPi config.ini to reuse")
    p.add_argument("--mic-rate", type=int, default=16000, help="mic sample rate: 16000, 32000 or 48000")
    p.add_argument("--out-rate", type=int, default=16000, help="speaker sample rate: 16000, 32000 or 48000")
    p.add_argument("--mic-device", default=None, help="input device: a number, or part of its name (see --list-devices)")
    p.add_argument("--out-device", default=None, help="output device: a number, or part of its name (see --list-devices)")
    p.add_argument("--barge-in", action="store_true",
                   help="keep the mic live while the skull talks (needs an echo-cancelling mic)")
    p.add_argument("--tail", type=float, default=0.4, help="seconds to keep the mic muted after the skull stops")
    p.add_argument("--no-servo", action="store_true", help="run without touching GPIO")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--jaw-test", action="store_true", help="step the jaw through its positions and exit")
    p.add_argument("--debug", action="store_true", help="print every server event type")
    args = p.parse_args(argv)
    args.persona_text = DEFAULT_PERSONA
    if os.path.isfile(args.persona):
        with open(args.persona, encoding="utf-8") as handle:
            args.persona_text = handle.read().strip() or DEFAULT_PERSONA
    return args


def main():
    args = parse_args()
    if args.list_devices:
        return list_devices()
    jaw = Jaw(load_chatterpi_config(args.config), enabled=not args.no_servo)
    if args.jaw_test:
        return jaw_test(jaw)
    SkullTalk(args, jaw).run()


if __name__ == "__main__":
    main()
