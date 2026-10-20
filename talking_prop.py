#!/usr/bin/env python3
"""talking_prop.py - live voice conversation for an animatronic prop with a moving jaw.

Mic -> OpenAI-style realtime speech-to-speech server (/v1/realtime) -> speaker,
with a jaw servo driven by the loudness of the reply audio, using the same
method and config.ini format as ChatterPi (https://github.com/ViennaMike/ChatterPi).

Files read from the script's own directory:
  config.ini     Servo calibration, levels and pins, in ChatterPi's format. Optional.
  persona.txt    The personality / instructions sent to the model. Optional.
  brave_key.txt  Brave Search API key. Optional; enables web search.
                 (The BRAVE_API_KEY environment variable works too.)

Run "python3 talking_prop.py --help" for options.
"""
import argparse
import base64
import configparser
import datetime
import html
import json
import os
import queue
import random
import re
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

SERVER_RATE = 16000  # the s2s server speaks 16 kHz mono 16-bit PCM both ways
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PERSONA = (
    "You are a friendly animatronic character. Keep every reply to one or two short "
    "spoken sentences. Never use lists, emoji or stage directions."
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


def load_eye_config(path):
    """Settings for an optional servo-driven eye from the [EYE_SERVOS] section."""
    cfg = configparser.ConfigParser()
    cfg.read(path)
    sec = cfg["EYE_SERVOS"] if cfg.has_section("EYE_SERVOS") else {}

    def num(key, default, kind=int):
        try:
            return kind(sec.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": str(sec.get("enabled", "OFF")).upper() == "ON",
        "pan_pin": num("pan_pin", 12),
        "pan_min": num("pan_min", 1000),
        "pan_max": num("pan_max", 2000),
        "pan_range": min(1.0, max(0.0, num("pan_range", 0.8, float))),
        "pan_reverse": str(sec.get("pan_reverse", "OFF")).upper() == "ON",
        "tilt_pin": num("tilt_pin", 13),
        "tilt_min": num("tilt_min", 1000),
        "tilt_max": num("tilt_max", 2000),
        "tilt_range": min(1.0, max(0.0, num("tilt_range", 0.5, float))),
        "tilt_reverse": str(sec.get("tilt_reverse", "OFF")).upper() == "ON",
        "min_hold": num("min_hold", 0.4, float),
        "max_hold": num("max_hold", 1.6, float),
    }


def load_song_config(path):
    """Settings for song mode from the [SONG] section (off unless a file is set)."""
    cfg = configparser.ConfigParser()
    cfg.read(path)
    sec = cfg["SONG"] if cfg.has_section("SONG") else {}
    base = os.path.dirname(os.path.abspath(path))
    # One file, or several separated by commas; each request plays the next one in order.
    files = []
    for name in str(sec.get("file", "")).split(","):
        name = name.strip()
        if name:
            files.append(name if os.path.isabs(name) else os.path.join(base, name))
    triggers = [t.strip().lower() for t in str(sec.get("triggers", "")).split(",") if t.strip()]
    try:
        button_pin = int(sec.get("button_pin", 0) or 0)
    except ValueError:
        button_pin = 0
    return {
        "files": files,
        "triggers": triggers,
        "button_pin": button_pin,
        "button_active_high": str(sec.get("button_active_high", "ON")).upper() == "ON",
        # ChatterPi drives the jaw from the right channel, so that's the default here too.
        "jaw_channel": "left" if str(sec.get("jaw_channel", "right")).lower() == "left" else "right",
        "output": "stereo" if str(sec.get("output", "mix")).lower() == "stereo" else "mix",
        "after": str(sec.get("after_song", "")).strip(),
    }


DEFAULT_PARROT_PERSONA = (
    "a sarcastic green parrot who rides on your shoulder, mocks you, repeats a word "
    "or two of what you said, and thinks it is the real brains of the outfit"
)


def load_parrot_config(path, jaw_settings):
    """Settings for the optional sidekick parrot from [PARROT]; its beak works like the jaw."""
    cfg = configparser.ConfigParser()
    cfg.read(path)
    sec = cfg["PARROT"] if cfg.has_section("PARROT") else {}

    def num(key, default, kind=int):
        try:
            return kind(sec.get(key, default))
        except (TypeError, ValueError):
            return default

    beak = dict(jaw_settings)  # same loudness levels as the jaw unless overridden
    beak.update({
        "jaw_pin": num("beak_pin", 21),
        "servo_min": num("servo_min", 1000),
        "servo_max": num("servo_max", 2000),
        "min_angle": num("min_angle", 0),
        "max_angle": num("max_angle", 90),
        "eyes": False,
    })
    for key in ("level1", "level2", "level3", "threshold", "style"):
        beak[key] = num(key, beak[key])
    return {
        "enabled": str(sec.get("enabled", "OFF")).upper() == "ON",
        "name": str(sec.get("name", "Polly")).strip() or "Polly",
        "persona": str(sec.get("persona", "")).strip() or DEFAULT_PARROT_PERSONA,
        "chance": min(1.0, max(0.0, num("chance", 0.5, float))),
        "speed": min(2.0, max(1.0, num("speed", 1.4, float))),
        "beak": beak,
    }


def speed_up(pcm, speed):
    """Play audio faster, which also raises its pitch: the parrot's voice from the Captain's."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if speed <= 1.0 or len(samples) < 2:
        return pcm
    new_len = max(1, int(len(samples) / speed))
    positions = np.linspace(0, len(samples) - 1, new_len)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.int16).tobytes()


def heard_trigger(transcript, triggers):
    """True if the visitor said one of the trigger phrases (ignoring case and punctuation)."""
    words = " ".join(re.sub(r"[^a-z0-9' ]+", " ", (transcript or "").lower()).split())
    return any(" ".join(t.split()) in words for t in triggers if t)


_PIN_FACTORY = []


def pin_factory():
    """pigpio for hardware-timed servo pulses if its daemon is running, else gpiozero's default."""
    if not _PIN_FACTORY:
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory

            _PIN_FACTORY.append(PiGPIOFactory())
        except Exception as exc:  # pigpiod not running or not installed
            print("pigpio unavailable (%s); using gpiozero's default pins. "
                  "Expect some servo jitter; see the pigpio section of the README." % exc)
            _PIN_FACTORY.append(None)
    return _PIN_FACTORY[0]


# --------------------------------------------------------------------------
# Web search (Brave Search API)
# --------------------------------------------------------------------------
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web for current or factual information: news, weather, sports scores, "
        "events, opening hours, or anything you are not sure about. Call it immediately instead of "
        "guessing, and never say you will look something up without calling it in the same turn. "
        "Read the results yourself and answer in one or two short sentences, in character. "
        "Never read out web addresses."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    },
}


TIME_TOOL = {
    "type": "function",
    "name": "get_current_time",
    "description": (
        "Get the current local date and time. Call it whenever you are asked the time, the date, "
        "the day of the week, or how long until something. Never guess the time."
    ),
    "parameters": {"type": "object", "properties": {}},
}

KID_SAFE_RULE = (
    "You are speaking with children and families. Keep everything friendly and age-appropriate. "
    "Never describe crime, death, violence, war, disasters or anything frightening from real life; "
    "if a question or a search result goes there, cheerfully change the subject."
)


# Replies where the model says it will look something up. Smaller models often say this
# without calling web_search; when they do, the client runs the search itself.
PROMISE_RE = re.compile(
    r"\b(i'?ll|i will|let me|lemme|gonna|going to|i'?m off to|i be)\b[^.!?]{0,50}?"
    r"\b(check|look|search|scour|find out|see what|dig|seek|hunt|ask)"
    r"|\b(hold fast|hold on|one moment|a tick|bear with me)\b",
    re.IGNORECASE,
)


def promised_to_search(reply):
    return bool(PROMISE_RE.search(reply or ""))


def current_time():
    now = datetime.datetime.now().astimezone()
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "timezone": now.tzname() or "",
    }


def load_brave_key(path):
    key = os.getenv("BRAVE_API_KEY", "").strip()
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            key = handle.read().strip()
    return key


def _plain(text):
    """Brave marks matches with <strong>; strip tags and HTML entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def brave_search(key, query, count=5, opener=urllib.request.urlopen):
    """Return a short, model-friendly list of results, or {"error": ...}."""
    url = BRAVE_URL + "?" + urllib.parse.urlencode({"q": query, "count": count, "safesearch": "strict"})
    request = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": key,
    })
    try:
        with opener(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": "search failed: %s" % exc}
    results = []
    for item in (data.get("web") or {}).get("results", [])[:count]:
        results.append({
            "title": _plain(item.get("title")),
            "snippet": _plain(item.get("description")),
            "source": urllib.parse.urlsplit(item.get("url", "")).netloc,
        })
    return {"query": query, "results": results} if results else {"query": query, "results": [], "note": "no results"}


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

        factory = pin_factory()
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

    @staticmethod
    def volume(samples):
        """Average loudness of a chunk, on the same scale as ChatterPi's levels."""
        return int(np.abs(samples.astype(np.int32)).mean()) if len(samples) else 0

    def target(self, samples):
        """Return the jaw angle for one chunk of int16 samples."""
        if len(samples) == 0:
            return self.closed
        volume = self.volume(samples)
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
# Eye
# --------------------------------------------------------------------------
class Eyes:
    """One eye on two servos (pan = side to side, tilt = up and down).

    While the prop talks, the eye glances to random spots, holding each for a moment;
    when it stops, the eye returns to centre and the servos are released so they don't buzz.
    Positions run from -1 (one end of the servo's travel) to +1 (the other), 0 is centre.
    """

    STEP = 0.03     # seconds between servo updates
    EASE = 0.35     # fraction of the remaining distance covered each update

    def __init__(self, settings, enabled=True, rng=None):
        self.s = settings
        self.rng = rng or random.Random()
        self.pan = self.tilt = None
        self.talking = threading.Event()
        self.stop = threading.Event()
        self.pos = [0.0, 0.0]
        self.target = [0.0, 0.0]
        self.drive = {"pan": True, "tilt": True}  # which servos get pulses (eye test can pick one)
        self.active = enabled and settings["enabled"]
        if not self.active:
            return
        from gpiozero import Servo

        factory = pin_factory()
        def servo(axis):
            low, high = sorted((settings[axis + "_min"], settings[axis + "_max"]))
            return Servo(settings[axis + "_pin"], initial_value=0, min_pulse_width=low / 1e6,
                         max_pulse_width=high / 1e6, pin_factory=factory)

        self.pan, self.tilt = servo("pan"), servo("tilt")
        self.release()

    def new_target(self):
        """A random spot within the allowed range, never too close to the last one."""
        while True:
            target = [self.rng.uniform(-1, 1) * self.s["pan_range"],
                      self.rng.uniform(-1, 1) * self.s["tilt_range"]]
            if abs(target[0] - self.target[0]) + abs(target[1] - self.target[1]) > 0.25 \
                    or self.s["pan_range"] + self.s["tilt_range"] < 0.25:
                return target

    def write(self):
        if self.pan is not None:
            pan = -self.pos[0] if self.s["pan_reverse"] else self.pos[0]
            tilt = -self.pos[1] if self.s["tilt_reverse"] else self.pos[1]
            if self.drive["pan"]:
                self.pan.value = max(-1.0, min(1.0, pan))
            if self.drive["tilt"]:
                self.tilt.value = max(-1.0, min(1.0, tilt))

    def release(self):
        if self.pan is not None:
            self.pan.value = None
            self.tilt.value = None

    def step_towards_target(self):
        moved = False
        for axis in (0, 1):
            gap = self.target[axis] - self.pos[axis]
            if abs(gap) > 0.01:
                self.pos[axis] += gap * self.EASE
                moved = True
            else:
                self.pos[axis] = self.target[axis]
        if moved:
            self.write()
        return moved

    def run(self):
        """Background loop: glance around while talking, centre and relax when not."""
        if not self.active:
            return
        next_glance = 0.0
        centred = True
        idle_since = time.monotonic()
        while not self.stop.is_set():
            now = time.monotonic()
            if self.talking.is_set():
                centred = False
                if now >= next_glance:
                    self.target = self.new_target()
                    next_glance = now + self.rng.uniform(self.s["min_hold"], self.s["max_hold"])
                self.step_towards_target()
                idle_since = now
            elif not centred:
                self.target = [0.0, 0.0]
                if not self.step_towards_target() and now - idle_since > 0.5:
                    self.release()
                    centred = True
            time.sleep(self.STEP)
        self.target = [0.0, 0.0]
        self.pos = [0.0, 0.0]
        self.write()
        time.sleep(0.2)
        self.release()

    def set_talking(self, on):
        (self.talking.set if on else self.talking.clear)()


EYE_TEST_MOVES = {
    "pan": ((-1, 0), (1, 0), (-1, 0), (1, 0), (0, 0)),
    "tilt": ((0, -1), (0, 1), (0, -1), (0, 1), (0, 0)),
    "both": ((-1, 0), (1, 0), (0, 0), (0, 1), (0, -1), (0, 0),
             (-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0)),
}


def eye_test(eyes, which="both"):
    """Move one or both eye servos to the ends of their range and back, then centre."""
    if not eyes.active:
        raise SystemExit("Eye servos are off. Set enabled = ON in the [EYE_SERVOS] section of config.ini.")
    s = eyes.s
    eyes.drive = {"pan": which in ("pan", "both"), "tilt": which in ("tilt", "both")}
    eyes.release()
    print("Testing %s: pin %s" % (which, ", ".join(
        "%s GPIO %d" % (axis, s[axis + "_pin"]) for axis in ("pan", "tilt") if eyes.drive[axis])))
    for pan, tilt in EYE_TEST_MOVES[which]:
        eyes.target = [pan * s["pan_range"], tilt * s["tilt_range"]]
        for _ in range(25):
            eyes.step_towards_target()
            time.sleep(eyes.STEP)
        time.sleep(0.3)
    eyes.release()


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
class TalkingProp:
    def __init__(self, args, jaw, eyes=None, song=None, parrot=None, beak=None, rng=None):
        self.args = args
        self.jaw = jaw
        self.eyes = eyes
        self.parrot = parrot if parrot and parrot.get("enabled") else None
        self.beak = beak                      # the parrot's servo (a Jaw on its own pin)
        self.rng = rng or random.Random()
        self.parrot_buffer = bytearray()      # sped-up parrot audio, played after the Captain's
        self.parrot_due = False               # the parrot gets the next turn
        self.parrot_asked = False             # we've asked the server for the parrot's line
        self.parrot_speaking = False          # the response in progress is the parrot's
        self.last_heard = ""                  # what the visitor last said (for the parrot)
        self.last_said = ""                   # what the Captain last said (for the parrot)
        self.song = song if song and song.get("files") else None
        self.song_index = 0                   # which song in the list plays next
        self.song_active = threading.Event()  # the song is playing
        self.song_stop = threading.Event()    # stop the song early
        self.song_requested = 0.0             # when a song was asked for (0 = not asked)
        self.song_intro_done = False          # the prop has finished its spoken intro
        self.pyaudio = None
        self.pa = None
        self.out_index = None
        self.speaker = None
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
        self.response_idle = threading.Event()  # set when no model response is in progress
        self.response_idle.set()
        self.send_lock = threading.Lock()
        self.last_user_text = ""
        self.reply_volumes = []    # loudness of each 20 ms of the current reply (--debug)
        self.called_tool = False   # the current response called a tool
        self.promised = False      # the current response said it would look something up

    # ---- audio device callbacks (run on PyAudio's thread) ----
    def on_mic(self, in_data, frame_count, time_info, status):
        try:
            self.mic_queue.put_nowait(in_data)
        except queue.Full:
            pass  # not connected or network stalled: drop audio rather than lag
        return (None, 0)  # 0 == pyaudio.paContinue

    def on_speaker(self, in_data, frame_count, time_info, status):
        want = (frame_count // self.out_factor) * 2  # bytes of 16 kHz audio
        mouth = self.jaw
        with self.play_lock:
            if self.play_buffer or not self.parrot_buffer:
                chunk = bytes(self.play_buffer[:want])
                del self.play_buffer[:want]
            else:  # the Captain has finished: now the parrot's line, with the beak moving
                chunk = bytes(self.parrot_buffer[:want])
                del self.parrot_buffer[:want]
                mouth = self.beak
        samples = np.frombuffer(chunk, dtype=np.int16)
        if len(samples):
            self.last_audio_time = time.monotonic()
            if mouth is not None:
                mouth.move(mouth.target(samples))
            if self.args.debug:
                self.reply_volumes.append(self.jaw.volume(samples))
        if len(chunk) < want:
            samples = np.concatenate([samples, np.zeros((want - len(chunk)) // 2, dtype=np.int16)])
        return (upsample(samples, self.out_factor).tobytes(), 0)

    def prop_is_talking(self):
        """True while reply audio or the song is playing, plus a short tail for room echo."""
        if self.song_active.is_set():
            return True
        with self.play_lock:
            pending = len(self.play_buffer) > 0 or len(self.parrot_buffer) > 0
        return pending or (time.monotonic() - self.last_audio_time) < self.args.tail

    # ---- websocket ----
    def send(self, message):
        """Send one JSON message; the mic thread and tool threads share the socket."""
        with self.send_lock:
            self.ws.send(json.dumps(message))

    def session_update(self):
        output = {"format": {"type": "audio/pcm", "rate": None}}
        if self.args.voice:
            output["voice"] = self.args.voice
        instructions = self.args.persona_text
        if self.args.kid_safe:
            instructions = KID_SAFE_RULE + "\n\n" + instructions
        now = current_time()
        instructions += ("\n\nRight now it is %s on %s (%s). When asked the time or date, give exactly "
                         "this, in your own words; never guess it." % (now["time"], now["date"], now["timezone"]))
        if self.parrot:
            instructions += ("\n\nYour parrot, %s, rides on your shoulder and sometimes speaks after you; "
                             "her lines are hers, not yours. You may talk about %s or grumble at her, but "
                             "never narrate what she does or the sounds she makes: never write things like "
                             "'%s squawks' or 'the parrot flaps', and never use the words 'squawk' or "
                             "'squawks'. Only say your own words."
                             % (self.parrot["name"], self.parrot["name"], self.parrot["name"]))
        if self.song and self.song["triggers"]:
            instructions += ("\n\nYou have a song you love to perform. If someone asks you to sing, never "
                             "refuse: reply with one short, eager line to introduce it, and the song "
                             "will start right after you speak.")
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": None},
                        "transcription": {"model": "gpt-4o-transcribe", "language": self.args.language},
                        "turn_detection": {"type": "server_vad", "interrupt_response": True},
                    },
                    "output": output,
                },
                "tools": [TIME_TOOL] + ([SEARCH_TOOL] if self.args.brave_key else []),
                "tool_choice": "auto",
            },
        }

    def on_open(self, ws):
        self.response_idle.set()
        self.send(self.session_update())
        self.connected.set()
        print("Connected to %s (voice=%s, web search %s)" % (
            self.args.url, self.args.voice or "server default", "on" if self.args.brave_key else "off"))

    def search_results_for(self, query):
        print("Search: %s" % query)
        result = brave_search(self.args.brave_key, query)
        if "error" in result:
            print("Search problem: %s" % result["error"])
        elif self.args.kid_safe:
            result["guidance"] = ("Children are listening. Mention only cheerful, family-friendly items; "
                                  "skip anything about crime, death, violence, war or disasters.")
        return result

    def fallback_search(self, query):
        """The model promised to look something up but didn't call web_search: do it for it."""
        result = self.search_results_for(query)
        self.response_idle.wait(15)
        if not self.connected.is_set():
            return
        note = ("[Web search results for the visitor's question %r, which you said you would look up. "
                "Answer it now from these, briefly and in character. Results: %s]" % (query, json.dumps(result)))
        try:
            self.send({"type": "conversation.item.create",
                       "item": {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": note}]}})
            self.send({"type": "response.create"})
        except Exception as exc:
            print("Could not return search results: %s" % exc)

    def run_tool(self, name, arguments, call_id):
        """Run a tool call from the model, then hand the result back and ask it to answer."""
        try:
            params = json.loads(arguments or "{}")
        except ValueError:
            params = {}
        if name == "get_current_time":
            result = current_time()
        elif name == "web_search" and self.args.brave_key and params.get("query"):
            result = self.search_results_for(params["query"])
        else:
            result = {"error": "unknown tool %r" % name}
        # The server takes tool results only after the response that asked for them is done.
        self.response_idle.wait(15)
        if not self.connected.is_set():
            return
        try:
            self.send({"type": "conversation.item.create",
                       "item": {"type": "function_call_output", "call_id": call_id,
                                "output": json.dumps(result)}})
            self.send({"type": "response.create"})
        except Exception as exc:
            print("Could not return search results: %s" % exc)

    def on_message(self, ws, message):
        try:
            event = json.loads(message)
        except ValueError:
            return
        kind = event.get("type", "")
        if kind in ("response.output_audio.delta", "response.audio.delta"):
            if self.song_active.is_set():
                return  # never talk over the song
            pcm = base64.b64decode(event.get("delta", ""))
            with self.play_lock:
                if self.parrot_speaking:
                    self.parrot_buffer.extend(speed_up(pcm, self.parrot["speed"]))
                else:
                    self.play_buffer.extend(pcm)
        elif kind == "input_audio_buffer.speech_started":
            if self.args.barge_in:  # someone spoke over the prop: stop it
                with self.play_lock:
                    del self.play_buffer[:]
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.last_user_text = (event.get("transcript") or "").strip()
            self.last_heard = self.last_user_text
            self.parrot_due = False  # a visitor spoke: they come first
            print("You:   %s" % self.last_user_text)
            if (self.song and not self.song_active.is_set() and not self.song_requested
                    and heard_trigger(self.last_user_text, self.song["triggers"])):
                print("Song requested; it starts after the intro.")
                self.song_requested = time.monotonic()
                self.song_intro_done = False
                self.last_user_text = ""  # no search fallback for this one
        elif kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
            reply = (event.get("transcript") or "").strip()
            if self.parrot_speaking:
                print("%-6s %s" % (self.parrot["name"] + ":", reply))
            else:
                print("Prop:  %s" % reply)
                self.promised = promised_to_search(reply)
                self.last_said = reply
        elif kind == "response.created":
            self.response_idle.clear()
            self.called_tool = self.promised = False
            self.parrot_speaking, self.parrot_asked = self.parrot_asked, False
        elif kind == "response.done":
            self.response_idle.set()
            if self.song_requested:
                self.song_intro_done = True
            searching = (self.promised and not self.called_tool and self.args.brave_key
                         and self.args.search_fallback and self.last_user_text)
            if searching:
                query, self.last_user_text = self.last_user_text, ""
                threading.Thread(target=self.fallback_search, args=(query,), daemon=True).start()
            elif (self.parrot and not self.parrot_speaking and not self.called_tool and self.last_said
                  and not self.song_requested and self.rng.random() < self.parrot["chance"]):
                self.parrot_due = True
            self.promised = False
            self.parrot_speaking = False
        elif kind == "response.function_call_arguments.done":
            self.called_tool = True
            threading.Thread(target=self.run_tool, daemon=True,
                             args=(event.get("name"), event.get("arguments"), event.get("call_id"))).start()
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
        """Forward mic audio; send silence while the prop talks unless barge-in is on."""
        peak, muted, sent, last_report = 0, False, 0, time.monotonic()
        while not self.stop.is_set():
            try:
                raw = self.mic_queue.get(timeout=0.5)
            except queue.Empty:
                if self.args.debug:
                    print("mic: no audio arriving from the input device")
                continue
            samples = downsample(np.frombuffer(raw, dtype=np.int16), self.mic_factor)
            if self.args.mic_gain != 1.0:
                samples = np.clip(samples.astype(np.float32) * self.args.mic_gain, -32768, 32767).astype(np.int16)
            if self.args.debug:
                if len(samples):
                    peak = max(peak, int(np.abs(samples.astype(np.int32)).max()))
                if time.monotonic() - last_report >= 1.0:
                    print("mic: peak %3d%%  sent=%d chunks%s%s" % (
                        peak * 100 // 32768, sent,
                        "" if self.connected.is_set() else "  (not connected)",
                        "  (muted while talking)" if muted else ""))
                    peak, sent, last_report = 0, 0, time.monotonic()
            if not self.connected.is_set():
                continue
            muted = self.song_active.is_set() or (not self.args.barge_in and self.prop_is_talking())
            if muted:
                samples = np.zeros_like(samples)
            message = {"type": "input_audio_buffer.append",
                       "audio": base64.b64encode(samples.tobytes()).decode("ascii")}
            try:
                self.send(message)
                sent += 1
            except Exception:
                self.connected.clear()

    def refresh_clock(self):
        """Resend the session every so often so the time in the instructions stays current.

        Small models often answer "what time is it?" from their instructions instead of calling
        get_current_time, so the instructions must not go stale. Only sent while idle.
        """
        last = time.monotonic()
        while not self.stop.wait(1.0):
            if time.monotonic() - last < self.args.clock_refresh:
                continue
            if not self.connected.is_set() or not self.response_idle.is_set() or self.prop_is_talking():
                continue
            try:
                self.send(self.session_update())
                last = time.monotonic()
            except Exception:
                pass

    def parrot_turn(self):
        """Ask the model for the parrot's line; its audio is sped up and moves the beak."""
        self.parrot_due = False
        name, said = self.parrot["name"], self.last_said
        prompt = ("[Parrot's turn. Speak as %s, %s. The visitor said: %r. You (the main character) just "
                  "said: %r. Give only the exact words %s says out loud: one short, sarcastic line under "
                  "15 words that pokes fun at what you said. A bird noise like 'Rawk!' is fine. No name, "
                  "no label, no narration and no description of what %s does.]"
                  % (name, self.parrot["persona"], self.last_heard, said, name, name))
        self.parrot_asked = True
        try:
            self.send({"type": "conversation.item.create",
                       "item": {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": prompt}]}})
            self.send({"type": "response.create"})
        except Exception:
            self.parrot_asked = False

    def request_song_now(self):
        """Button press: start the song right away, or stop it if it's playing."""
        if self.song_active.is_set():
            print("Song stopped by button.")
            self.song_stop.set()
        elif self.song:
            print("Song started by button.")
            with self.play_lock:
                del self.play_buffer[:]  # cut off anything the prop was saying
                del self.parrot_buffer[:]
            self.parrot_due = False
            self.song_requested = time.monotonic()
            self.song_intro_done = True

    def song_ready(self):
        """A requested song can start once the spoken intro is over (or after 8 s regardless)."""
        if not self.song_requested or self.song_active.is_set():
            return False
        waited = time.monotonic() - self.song_requested
        with self.play_lock:
            still_speaking = len(self.play_buffer) > 0
        return not still_speaking and ((self.song_intro_done and self.response_idle.is_set()) or waited > 8)

    def open_speaker(self):
        pyaudio = self.pyaudio
        self.speaker = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.out_rate, output=True,
                                    output_device_index=self.out_index,
                                    frames_per_buffer=int(self.args.out_rate * 0.02),
                                    stream_callback=self.on_speaker)

    def play_song(self):
        """Play the song WAV like ChatterPi: one channel drives the jaw, the speaker gets the music."""
        import wave

        pyaudio = self.pyaudio
        song = self.song
        self.song_requested = 0.0
        path = song["files"][self.song_index % len(song["files"])]
        self.song_index = (self.song_index + 1) % len(song["files"])  # next request plays the next song
        try:
            wav = wave.open(path, "rb")
        except (OSError, wave.Error) as exc:
            print("Can't play the song: %s" % exc)
            self.song_active.clear()
            return
        if wav.getsampwidth() != 2:
            print("Can't play the song: it must be 16-bit PCM WAV. Convert it with: "
                  "ffmpeg -i in.wav -c:a pcm_s16le -ar 44100 song.wav")
            wav.close()
            self.song_active.clear()
            return
        channels, rate = wav.getnchannels(), wav.getframerate()
        jaw_index = 1 if (channels == 2 and song["jaw_channel"] == "right") else 0
        out_channels = channels
        print("Song: %s (%d ch, %d Hz, %.0f s)" % (os.path.basename(path), channels, rate,
                                                  wav.getnframes() / float(rate)))
        self.song_stop.clear()
        self.song_active.set()
        with self.play_lock:
            del self.play_buffer[:]
        if self.speaker is not None:  # the sound card can only be opened once: swap streams
            self.speaker.stop_stream()
            self.speaker.close()
            self.speaker = None
        finished = threading.Event()

        def callback(in_data, frame_count, time_info, status):
            data = wav.readframes(frame_count)
            frames = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
            if len(frames) == 0 or self.song_stop.is_set():
                finished.set()
                return (b"\x00" * frame_count * out_channels * 2, pyaudio.paComplete)
            self.jaw.move(self.jaw.target(frames[:, jaw_index]))
            if channels == 2 and song["output"] == "mix":  # one speaker: hear voice and music together
                mono = (frames.astype(np.int32).sum(axis=1) // 2).astype(np.int16)
                frames = np.repeat(mono[:, None], 2, axis=1)
            out = frames.tobytes()
            if len(frames) < frame_count:
                out += b"\x00" * (frame_count - len(frames)) * out_channels * 2
                finished.set()
                return (out, pyaudio.paComplete)
            return (out, pyaudio.paContinue)

        stream = None
        try:
            stream = self.pa.open(format=pyaudio.paInt16, channels=out_channels, rate=rate, output=True,
                                  output_device_index=self.out_index, frames_per_buffer=int(rate * 0.02),
                                  stream_callback=callback)
            while not finished.is_set() and not self.stop.is_set():
                finished.wait(0.2)
            time.sleep(0.2)
        except Exception as exc:
            print("Song playback problem: %s" % exc)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            wav.close()
            if not self.stop.is_set() and self.speaker is None and self.args.url:
                self.open_speaker()
            self.last_audio_time = time.monotonic()
            self.song_active.clear()
            print("Song finished; back to conversation.")
        if song["after"] and self.connected.is_set() and not self.stop.is_set():
            try:
                self.send({"type": "conversation.item.create",
                           "item": {"type": "message", "role": "user",
                                    "content": [{"type": "input_text", "text": song["after"]}]}})
                self.send({"type": "response.create"})
            except Exception:
                pass

    def watch_speaking(self):
        """Light the eyes (if wired) while talking and relax the servo afterwards."""
        while not self.stop.is_set():
            if (self.parrot_due and self.connected.is_set() and self.response_idle.is_set()
                    and not self.prop_is_talking() and not self.song_requested):
                self.parrot_turn()
            if self.song and self.song_ready():
                self.song_requested = 0.0
                self.song_active.set()  # claim it now so it can't start twice
                threading.Thread(target=self.play_song, daemon=True).start()
            talking = self.prop_is_talking()
            if talking and not self.is_speaking:
                self.jaw.speaking(True)
                if self.eyes is not None:
                    self.eyes.set_talking(True)
            elif not talking and self.is_speaking:
                if self.args.debug and self.reply_volumes:
                    vols = sorted(self.reply_volumes)
                    s = self.jaw.s
                    print("jaw: reply loudness median %d, loud parts %d, peak %d  (levels %d / %d / %d)" % (
                        vols[len(vols) // 2], vols[int(len(vols) * 0.9)], vols[-1],
                        s["level1"], s["level2"], s["level3"]))
                    self.reply_volumes = []
                self.jaw.speaking(False)
                if self.eyes is not None:
                    self.eyes.set_talking(False)
                self.jaw.rest()
                if self.beak is not None:
                    self.beak.rest()
            self.is_speaking = talking
            time.sleep(0.05)

    def start_audio(self):
        import pyaudio

        self.pyaudio = pyaudio
        self.pa = pyaudio.PyAudio()
        self.out_index = find_device(self.pa, self.args.out_device, need_input=False)

    def start_button(self):
        """Song button on a GPIO: press to start the song, press again to stop it."""
        if not self.song or not self.song["button_pin"] or self.args.no_servo:
            return None
        from gpiozero import Button

        button = Button(self.song["button_pin"], pull_up=not self.song["button_active_high"],
                        bounce_time=0.1, pin_factory=pin_factory())
        button.when_pressed = self.request_song_now
        print("Song button on GPIO %d" % self.song["button_pin"])
        return button

    def run(self):
        self.start_audio()
        pyaudio, pa = self.pyaudio, self.pa
        mic_index = find_device(pa, self.args.mic_device, need_input=True)
        mic = pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.mic_rate, input=True,
                      input_device_index=mic_index,
                      frames_per_buffer=int(self.args.mic_rate * 0.04), stream_callback=self.on_mic)
        self.open_speaker()
        button = self.start_button()  # noqa: F841 (kept alive for its callback)
        if self.song:
            print("Songs ready, in order: %s%s" % (", ".join(os.path.basename(f) for f in self.song["files"]),
                                        ", say: " + " / ".join(self.song["triggers"]) if self.song["triggers"] else ""))
        workers = [self.run_socket, self.send_mic, self.watch_speaking, self.refresh_clock]
        if self.eyes is not None:
            workers.append(self.eyes.run)
        threads = [threading.Thread(target=fn, daemon=True) for fn in workers]
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
            self.song_stop.set()
            mic.stop_stream()
            mic.close()
            if self.speaker is not None:
                self.speaker.stop_stream()
                self.speaker.close()
            time.sleep(0.3)
            pa.terminate()
            self.jaw.rest()
            if self.eyes is not None:
                self.eyes.stop.set()
                time.sleep(0.5)


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
    p = argparse.ArgumentParser(description="Live voice conversation for an animatronic prop.")
    p.add_argument("--url", default=os.getenv("TALKING_PROP_URL", ""),
                   help="realtime websocket URL of the s2s server, e.g. ws://192.168.1.50:8765/v1/realtime "
                        "(or set TALKING_PROP_URL)")
    p.add_argument("--token", default=os.getenv("TALKING_PROP_TOKEN", "DUMMY"),
                   help="bearer token, if the server wants one")
    p.add_argument("--voice", default=os.getenv("TALKING_PROP_VOICE", ""), help="voice name known to the s2s server")
    p.add_argument("--language", default="en", help="transcription language")
    p.add_argument("--persona", default=os.path.join(HERE, "persona.txt"), help="file with the personality text")
    p.add_argument("--config", default=os.path.join(HERE, "config.ini"), help="ChatterPi config.ini to reuse")
    p.add_argument("--mic-rate", type=int, default=16000, help="mic sample rate: 16000, 32000 or 48000")
    p.add_argument("--out-rate", type=int, default=16000, help="speaker sample rate: 16000, 32000 or 48000")
    p.add_argument("--mic-device", default=None, help="input device: a number, or part of its name (see --list-devices)")
    p.add_argument("--out-device", default=None, help="output device: a number, or part of its name (see --list-devices)")
    p.add_argument("--barge-in", action="store_true",
                   help="keep the mic live while the prop talks (needs an echo-cancelling mic)")
    p.add_argument("--mic-gain", type=float, default=1.0,
                   help="multiply the mic signal, e.g. 3 for a quiet mic (try the mixer level first)")
    p.add_argument("--tail", type=float, default=0.4, help="seconds to keep the mic muted after the prop stops")
    p.add_argument("--no-servo", action="store_true", help="run without touching GPIO")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--jaw-test", action="store_true", help="step the jaw through its positions and exit")
    p.add_argument("--parrot-test", action="store_true", help="step the parrot's beak through its positions and exit")
    p.add_argument("--sing", action="store_true",
                   help="play the [SONG] file once with the jaw and eye moving, then exit (no server needed)")
    p.add_argument("--eye-test", nargs="?", const="both", choices=("pan", "tilt", "both"),
                   help="move the eye servos through their range and exit; "
                        "add pan or tilt to test just one (default both)")
    p.add_argument("--brave-key-file", default=os.path.join(HERE, "brave_key.txt"),
                   help="file holding the Brave Search API key (or set BRAVE_API_KEY)")
    p.add_argument("--no-search", action="store_true", help="turn web search off even if a key is present")
    p.add_argument("--no-search-fallback", dest="search_fallback", action="store_false",
                   help="don't search on the model's behalf when it says it will look something up but doesn't")
    p.add_argument("--kid-safe", action="store_true",
                   help="keep answers and search results family-friendly (for props around children)")
    p.add_argument("--clock-refresh", type=float, default=60.0,
                   help="seconds between updates of the time given to the model (default 60)")
    p.add_argument("--debug", action="store_true", help="print server events and a mic level meter once a second")
    args = p.parse_args(argv)
    args.brave_key = "" if args.no_search else load_brave_key(args.brave_key_file)
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
    eyes = Eyes(load_eye_config(args.config), enabled=not args.no_servo)
    if args.eye_test:
        return eye_test(eyes, args.eye_test)
    parrot = load_parrot_config(args.config, jaw.s)
    beak = None
    if parrot["enabled"] or args.parrot_test:
        beak = Jaw(parrot["beak"], enabled=not args.no_servo)
    if args.parrot_test:
        return jaw_test(beak)
    song = load_song_config(args.config)
    for missing in [f for f in song["files"] if not os.path.isfile(f)]:
        print("Song file not found, skipping: %s" % missing)
    song["files"] = [f for f in song["files"] if os.path.isfile(f)]
    if not song["files"]:
        song = None
    if args.sing:
        if not song:
            raise SystemExit("No song. Set file = ... in the [SONG] section of config.ini.")
        return sing_once(args, jaw, eyes if eyes.active else None, song)
    if not args.url:
        raise SystemExit("No server URL. Pass --url ws://<server>:8765/v1/realtime or set TALKING_PROP_URL.")
    if parrot["enabled"]:
        print("Parrot %s on GPIO %d, chimes in after about %d%% of replies"
              % (parrot["name"], parrot["beak"]["jaw_pin"], parrot["chance"] * 100))
    TalkingProp(args, jaw, eyes if eyes.active else None, song, parrot, beak).run()


def sing_once(args, jaw, eyes, song):
    """--sing: perform the song without a server, to check levels, channels and movement."""
    prop = TalkingProp(args, jaw, eyes, song)
    prop.start_audio()
    workers = [prop.watch_speaking] + ([eyes.run] if eyes is not None else [])
    for fn in workers:
        threading.Thread(target=fn, daemon=True).start()
    button = prop.start_button()  # noqa: F841 (press to stop early)
    try:
        prop.play_song()
    except KeyboardInterrupt:
        prop.song_stop.set()
    finally:
        prop.stop.set()
        jaw.rest()
        time.sleep(0.5)
        prop.pa.terminate()


if __name__ == "__main__":
    main()
